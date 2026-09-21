//! `FFTDF` — plane-wave density fitting on the FFT box (plans 11-05 / 11-08).
//!
//! Ports `pyscf/pbc/df/fft.py:40-80` (`get_nuc`), `:82-178` (`get_pp`) and
//! `:185-405` (the `FFTDF` class).
//!
//! # Deviation from upstream's `get_pp` (documented, deliberate)
//!
//! `fft.py`'s `get_pp` evaluates the NON-LOCAL half in reciprocal space through
//! `ft_ao.ft_ao`, the McMurchie-Davidson planewave AO transform that
//! PBC-MASTER-PLAN schedules for Phase 13. Phase 10 already shipped the SAME
//! quantity in real space — [`pyscf_pbc_gto::pseudo::get_pp_nl`], via
//! `intor_cross` against the projector fake-cell — and gated it against
//! upstream at 1.9e-15 on diamond. This port therefore assembles
//!
//! ```text
//! V_pp(k) = ifft(-sum_a SI[a] * vlocG[a])  +  V_nl(k)
//! ```
//!
//! using the FFT for the local half (identical to upstream) and the Phase-10
//! real-space route for the non-local half. Both are the same operator; the
//! only difference is which quadrature evaluates it, and the real-space one is
//! the more accurate of the two (it is exact in the basis, with no planewave
//! truncation). `tests/fftdf.rs` pins the assembled `V_pp` against upstream.
//!
//! # The AO cache
//!
//! `eval_ao_kpts` over `ngrids = mesh.product()` points is the single most
//! expensive non-FFT step, and neither the grid nor the k-points move during an
//! SCF. Upstream re-evaluates on every `aoR_loop`; this port memoises the AO
//! table per k-point list, bounded by [`Fftdf::max_memory`]. Cached values are
//! bit-identical to a fresh evaluation — it is the same function of the same
//! inputs.

use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use pyscf_algebra::{CTensor, openblas_emu};
use pyscf_pbc_gto::{
    Cell, CoulGArgs, ExxDiv, UniformGrids, eval_ao_kpts, eval_ao_kpts_upstream, get_coulg,
    get_coulg_at_gv, get_gv, get_si, is_zero,
};
use pyscf_pbc_tools::{ifft, ifft_upstream};

use crate::df_jk::KMats;
use crate::error::PbcDfError;
use crate::traits::{JkOpts, JkResult, PeriodicDf};
use crate::zlinalg::{forder_to_c, zadd_assign};

/// AO values on the uniform grid, one `(nao, ngrids)` ROW-MAJOR block per
/// k-point.
///
/// That layout is `eval_ao_kpts`'s native one (its `(ngrids, nao)` F-order per
/// component is the same buffer) and it is also upstream's `ao2T`/`ao1T`, so no
/// transpose happens anywhere on the J/K path.
#[derive(Debug, Clone, PartialEq)]
pub struct AoKpts {
    /// AO count.
    pub nao: usize,
    /// Grid-point count.
    pub ngrids: usize,
    /// `aot[k][mu * ngrids + g]`. Shared (`Arc`) so that a band table that is
    /// a subset of the sampling table — S-08 — points at the sampling table's
    /// blocks instead of copying or re-evaluating them.
    pub aot: Vec<Arc<CTensor>>,
}

impl AoKpts {
    /// The `(nao, ngrids)` block at k-point `k`.
    pub fn at(&self, k: usize) -> &CTensor {
        &self.aot[k]
    }
}

/// `FFTDF(cell, kpts)` — `fft.py:185-405`.
#[derive(Debug)]
pub struct Fftdf {
    /// The cell.
    pub cell: Cell,
    /// Sampling k-points. Empty means the single gamma point.
    pub kpts: Vec<[f64; 3]>,
    /// FFT mesh; defaults to `cell.mesh`. Assigning to it directly leaves
    /// [`Fftdf::grids`] and the AO cache describing the OLD mesh — use
    /// [`Fftdf::set_mesh`], which is upstream's `mydf.mesh = ...` (its `grids`
    /// is a property recomputed on every read).
    pub mesh: [usize; 3],
    /// The uniform quadrature grid on `mesh`.
    pub grids: UniformGrids,
    /// Memory budget in MB, used to size the `get_k_kpts` AO block and to cap
    /// the AO cache. Upstream's `mydf.max_memory`.
    pub max_memory: f64,
    /// Cached `(nao, ngrids)` AO tables, keyed by the k-point list.
    ao_cache: Mutex<HashMap<Vec<[u64; 3]>, Arc<AoKpts>>>,
    /// W-01: `get_coulG(dk)` keyed on the RAW `dk` bits plus `omega` and the
    /// exxdiv actually applied INSIDE the k-pair loop (never `Ewald` on the
    /// energy path; passed through on the gradient path — see
    /// [`Fftdf::coulg_and_expmikr`]).
    ///
    /// 18-04 (D-PBC-31 clause 9) re-keyed this on the wrapped k-difference
    /// class ([`crate::ao_cache::kdiff_index`]) on the premise that `coulG` is
    /// class-invariant. It is not, per grid point: `dk -> dk + b` reorders the
    /// `G + k` array, so the class key served `+b1/2`'s table to `-b1/2` and
    /// moved `KRHF` on diamond `[2,1,1]` by 0.184 Ha
    /// (`kscf::supercell_equivalence_holds`, found and reverted in Phase 20 —
    /// `20-pbc-python-bindings/measurements/kscf-supercell-regression.md`).
    /// Re-introducing the class key needs the per-class permutation, not a
    /// shared table.
    coulg_cache: Mutex<HashMap<CoulgKey, Arc<Vec<f64>>>>,
    /// The `expmikr(dk) = exp(-i dk·r)` phase tables, keyed on the RAW `dk`
    /// bits. Unlike `coulG`, the phase depends on the unwrapped
    /// representative, so it keeps its own map. These builds are `O(ngrids)`
    /// trigonometry, not `get_coulG` calls, and are not counted below.
    expmikr_cache: Mutex<HashMap<[u64; 3], Option<Arc<CTensor>>>>,
    /// How many `get_coulG` builds the W-01 cache has performed on this
    /// builder. The 18-04 gate asserts this reads `nkpts` (not `nkpts²`) over
    /// a full `get_k_e1_kpts` call — the cache is counted, not assumed. Not
    /// cleared by [`Fftdf::reset`]; use [`Fftdf::reset_coulg_build_count`].
    coulg_builds: AtomicUsize,
}

/// One cached `coulG` + phase pair. The two halves carry separate `Arc`s
/// because they are keyed differently (class index vs raw `dk`) — cloning the
/// tuple per pair would deep-copy `ngrids` doubles on every one of the `Nk^2`
/// pairs, which is exactly the per-pair work W-01 exists to stop paying
/// (S-04: borrow through the `Arc`s).
#[derive(Debug, Clone)]
pub struct CoulgEntry {
    /// `get_coulG(dk)`, shared by every pair in the k-difference class.
    pub coulg: Arc<Vec<f64>>,
    /// `expmikr(dk)`, `None` on the diagonal — specific to the raw `dk`.
    pub expmikr: Option<Arc<CTensor>>,
}

/// `(dk.to_bits(), omega.to_bits(), exxdiv)` — see
/// [`Fftdf::coulg_cache`].
type CoulgKey = ([u64; 3], Option<u64>, Option<ExxDiv>);

/// Upstream's `lib.param.MAX_MEMORY` default, in MB, overridable through
/// `PYSCF_MAX_MEMORY` (the same variable the molecular crates read).
fn default_max_memory() -> f64 {
    std::env::var("PYSCF_MAX_MEMORY")
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| *v > 0.0)
        .unwrap_or(4000.0)
}

fn kpt_key(kpts: &[[f64; 3]]) -> Vec<[u64; 3]> {
    kpts.iter()
        .map(|k| [k[0].to_bits(), k[1].to_bits(), k[2].to_bits()])
        .collect()
}

/// When [`Fftdf::local_vmat`] takes the K-14f fused route, from
/// `PYSCF_PBC_HCORE_FUSE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HcoreFuse {
    /// `0` — never. The pre-K-14f route: evaluate the whole AO table, read it
    /// back, cache it, reduce on the host. The bit-identity reference arm.
    Never,
    /// Unset (the default), or `auto` — fuse exactly when the table would NOT
    /// be admitted to the AO cache. See [`Fftdf::local_vmat`] for why that is
    /// the right predicate and not a memory threshold of its own.
    Auto,
    /// `1` — always, whatever the cache would have done. The measurement arm,
    /// and the setting for a one-shot `get_hcore` on a memory-tight machine.
    Always,
}

fn hcore_fuse_mode() -> HcoreFuse {
    match std::env::var("PYSCF_PBC_HCORE_FUSE").as_deref() {
        Ok("0") => HcoreFuse::Never,
        Ok("1") => HcoreFuse::Always,
        _ => HcoreFuse::Auto,
    }
}

impl Fftdf {
    /// Build an `FFTDF` for `cell` at `kpts` (empty = gamma), using
    /// `cell.mesh`.
    ///
    /// # Errors
    /// [`PbcDfError::Core`] when the cell has no mesh, or the grid cannot be
    /// built.
    pub fn new(cell: Cell, kpts: &[[f64; 3]]) -> Result<Self, PbcDfError> {
        let mesh = cell.try_mesh()?;
        Self::with_mesh(cell, kpts, mesh)
    }

    /// [`Fftdf::new`] with an explicit mesh — upstream's `mydf.mesh = ...`.
    ///
    /// # Errors
    /// As [`Fftdf::new`].
    pub fn with_mesh(cell: Cell, kpts: &[[f64; 3]], mesh: [usize; 3]) -> Result<Self, PbcDfError> {
        let grids = UniformGrids::build(&cell, Some(mesh))?;
        let kpts = if kpts.is_empty() {
            vec![[0.0; 3]]
        } else {
            kpts.to_vec()
        };
        Ok(Self {
            cell,
            kpts,
            mesh,
            grids,
            max_memory: default_max_memory(),
            ao_cache: Mutex::new(HashMap::new()),
            coulg_cache: Mutex::new(HashMap::new()),
            expmikr_cache: Mutex::new(HashMap::new()),
            coulg_builds: AtomicUsize::new(0),
        })
    }

    /// `mydf.mesh = ...` — rebuild the grid and drop the AO cache.
    ///
    /// # Errors
    /// Propagates the grid construction.
    pub fn set_mesh(&mut self, mesh: [usize; 3]) -> Result<(), PbcDfError> {
        self.grids = UniformGrids::build(&self.cell, Some(mesh))?;
        self.mesh = mesh;
        self.reset();
        Ok(())
    }

    /// `ngrids = mesh.product()`.
    pub fn ngrids(&self) -> usize {
        self.grids.size()
    }

    /// The quadrature weight `vol / ngrids`.
    pub fn weight(&self) -> f64 {
        self.grids.weight()
    }

    /// The `(nao, ngrids)` AO table at `kpts`, from the cache when possible.
    ///
    /// # Errors
    /// Propagates [`eval_ao_kpts`].
    pub fn ao_kpts(&self, kpts: &[[f64; 3]]) -> Result<Arc<AoKpts>, PbcDfError> {
        let key = kpt_key(kpts);
        if let Ok(c) = self.ao_cache.lock() {
            if let Some(v) = c.get(&key) {
                return Ok(Arc::clone(v));
            }
        }
        // S-08: a k-list that is a (bitwise) subset of the sampling list —
        // what every k-symmetric driver passes as `kpts_band = kpts_ibz` — is
        // served from the sampling table. K-08 accumulates each k
        // independently (`out[k·n+p] += phase_k · ao[p]`, the same AO block
        // and the same phase whichever list it is launched with), so the
        // block is bit-identical; the `Arc` makes it free of copies too.
        // `PYSCF_PBC_BAND_AO_REUSE=0` restores the separate evaluation.
        let reuse = !std::env::var("PYSCF_PBC_BAND_AO_REUSE").is_ok_and(|v| v == "0");
        if reuse && kpts.len() < self.kpts.len() {
            let same = |a: &[f64; 3], b: &[f64; 3]| {
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
            };
            let map: Option<Vec<usize>> = kpts
                .iter()
                .map(|b| self.kpts.iter().position(|k| same(k, b)))
                .collect();
            if let Some(map) = map {
                let full = self.ao_kpts(&self.kpts.clone())?;
                let block = Arc::new(AoKpts {
                    nao: full.nao,
                    ngrids: full.ngrids,
                    aot: map.iter().map(|&k| Arc::clone(&full.aot[k])).collect(),
                });
                if let Ok(mut c) = self.ao_cache.lock() {
                    c.insert(key, Arc::clone(&block));
                }
                return Ok(block);
            }
        }
        let out = eval_ao_kpts(&self.cell, "GTOval_sph", &self.grids.coords, kpts)?;
        debug_assert_eq!(out.comp, 1, "the LDA AO path evaluates one component");
        let block = Arc::new(AoKpts {
            nao: out.nao,
            ngrids: out.ngrids,
            aot: out.kaos.into_iter().map(Arc::new).collect(),
        });
        // 16 bytes per complex entry; keep the cache under a quarter of the
        // memory budget so the J/K scratch still fits.
        if self.ao_table_fits_cache(kpts.len()) {
            if let Ok(mut c) = self.ao_cache.lock() {
                c.insert(key, Arc::clone(&block));
            }
        }
        Ok(block)
    }

    /// How many AO tables the cache holds.
    ///
    /// Exposed for the K-14f gate: the fused local contraction must leave the
    /// cache COLD (it evaluates and reduces in one pass and keeps nothing),
    /// while the host route caches the table it built. A test cannot tell
    /// those apart from the returned matrices, which are bit-identical.
    pub fn ao_cache_len(&self) -> usize {
        self.ao_cache.lock().map_or(0, |c| c.len())
    }

    /// Drop the AO cache and the W-01 `coulG`/`expmikr` caches — call after
    /// mutating `cell` or `mesh`. The [`Fftdf::coulg_build_count`] counter is
    /// NOT cleared here; it has its own reset so a gate can count builds
    /// across several calls.
    pub fn reset(&self) {
        if let Ok(mut c) = self.ao_cache.lock() {
            c.clear();
        }
        if let Ok(mut c) = self.coulg_cache.lock() {
            c.clear();
        }
        if let Ok(mut c) = self.expmikr_cache.lock() {
            c.clear();
        }
    }

    /// How many `get_coulG` builds the W-01 cache has performed (cache misses
    /// that computed, not lookups). The 18-04 clause-9 gate asserts this.
    pub fn coulg_build_count(&self) -> usize {
        self.coulg_builds.load(Ordering::Relaxed)
    }

    /// Zero [`Fftdf::coulg_build_count`].
    pub fn reset_coulg_build_count(&self) {
        self.coulg_builds.store(0, Ordering::Relaxed);
    }

    /// `get_coulG(dk)` and the phase table `expmikr(dk) = exp(-i dk.r)` on
    /// `self.grids.coords`, memoised — `fft_jk.py`'s `get_k_kpts` rebuilds
    /// both from scratch on every one of the `Nk^2` `(k1, k2)` pairs even
    /// though neither depends on the density matrix or on which pair produced
    /// this particular `dk` (W-01, §2.4 of KRKS-OPTIMISATION-PLAN.md).
    ///
    /// The two halves are keyed differently (D-PBC-31 clause 9): `coulG` on
    /// the k-difference index (the wrapped class — `Nk^2 → Nk` builds), the
    /// phase on the raw `dk` (it depends on the representative). Both are
    /// still computed from the raw `dk`, exactly as before, so a hit returns
    /// the same bytes the pre-18-04 cache returned.
    ///
    /// `exxdiv` here is the value ACTUALLY passed to `get_coulG` inside the
    /// pair loop. `fft_jk::get_k_kpts` maps `Some(ExxDiv::Ewald) | None` to
    /// `None` before calling this (the Ewald probe-charge correction is
    /// applied once, after the loop, at `G+k = 0`); the gradient
    /// `get_k_e1_kpts` passes `exxdiv` through unchanged (including `Ewald` —
    /// upstream's gradient tail has no `_ewald_exxdiv_for_G0`), so `exxdiv`
    /// here CAN be `Some(ExxDiv::Ewald)` on the gradient route. The key
    /// carries it either way, so the two routes never share an entry.
    ///
    /// # Errors
    /// Propagates [`get_coulg`].
    pub fn coulg_and_expmikr(
        &self,
        dk: [f64; 3],
        omega: Option<f64>,
        exxdiv: Option<ExxDiv>,
        kpts: &[[f64; 3]],
        gv: &[[f64; 3]],
    ) -> Result<Arc<CoulgEntry>, PbcDfError> {
        let raw_key = [dk[0].to_bits(), dk[1].to_bits(), dk[2].to_bits()];
        // Keyed on the RAW `dk` bits, not `kdiff_index`: `get_coulG` is NOT
        // invariant per grid point under `dk -> dk + b` (the reciprocal shift
        // reorders the `G + k` array), so a wrapped-class key served `+b1/2`'s
        // table to `-b1/2` and moved KRHF on diamond [2,1,1] by 0.184 Ha
        // (20-VERIFICATION, measurements/kscf-supercell-regression.md).
        let class_key: CoulgKey = (raw_key, omega.map(f64::to_bits), exxdiv);
        if let Ok(c) = self.coulg_cache.lock()
            && let Ok(e) = self.expmikr_cache.lock()
            && let (Some(coulg), Some(expmikr)) = (c.get(&class_key), e.get(&raw_key))
        {
            return Ok(Arc::new(CoulgEntry {
                coulg: Arc::clone(coulg),
                expmikr: expmikr.clone(),
            }));
        }
        let coulg = if let Ok(c) = self.coulg_cache.lock() {
            if let Some(v) = c.get(&class_key) {
                Arc::clone(v)
            } else {
                drop(c);
                let built = get_coulg(
                    &self.cell,
                    CoulGArgs {
                        k: dk,
                        exxdiv,
                        kpts: Some(kpts),
                        mesh: Some(self.mesh),
                        gv: Some(gv),
                        wrap_around: true,
                        omega,
                    },
                )?;
                let built = Arc::new(built);
                self.coulg_builds.fetch_add(1, Ordering::Relaxed);
                if let Ok(mut c) = self.coulg_cache.lock() {
                    c.insert(class_key, Arc::clone(&built));
                }
                built
            }
        } else {
            Arc::new(get_coulg(
                &self.cell,
                CoulGArgs {
                    k: dk,
                    exxdiv,
                    kpts: Some(kpts),
                    mesh: Some(self.mesh),
                    gv: Some(gv),
                    wrap_around: true,
                    omega,
                },
            )?)
        };
        let expmikr: Option<Arc<CTensor>> = if let Ok(e) = self.expmikr_cache.lock() {
            if let Some(v) = e.get(&raw_key) {
                v.clone()
            } else {
                drop(e);
                let built = self.build_expmikr(dk).map(Arc::new);
                if let Ok(mut e) = self.expmikr_cache.lock() {
                    e.insert(raw_key, built.clone());
                }
                built
            }
        } else {
            self.build_expmikr(dk).map(Arc::new)
        };
        Ok(Arc::new(CoulgEntry { coulg, expmikr }))
    }

    /// `expmikr(dk) = exp(-i dk.r)` on the grid, `None` when `dk` is zero —
    /// the `is_zero(kpt1-kpt2)` branch (`fft_jk.py:386-387`), which is also
    /// what lets the gradient loop skip the whole-array phase multiply on the
    /// diagonal (D-PBC-31 clause 11).
    fn build_expmikr(&self, dk: [f64; 3]) -> Option<CTensor> {
        if is_zero(&dk) {
            None
        } else {
            let ngrids = self.grids.coords.len();
            let mut re = vec![0.0_f64; ngrids];
            let mut im = vec![0.0_f64; ngrids];
            for (g, r) in self.grids.coords.iter().enumerate() {
                let ph = -(r[0] * dk[0] + r[1] * dk[1] + r[2] * dk[2]);
                re[g] = ph.cos();
                im[g] = ph.sin();
            }
            Some(CTensor::from_planes(re, im))
        }
    }

    /// Whether a `(nkpts, nao, ngrids)` complex AO table is small enough for
    /// the AO cache to keep — 16 bytes per entry, under a quarter of the
    /// memory budget so the J/K scratch still fits.
    ///
    /// [`Fftdf::ao_kpts`] decides admission with it and [`Fftdf::local_vmat`]
    /// routes on it, so the two cannot drift apart: the fused contraction
    /// engages exactly on the tables the cache was going to refuse.
    fn ao_table_fits_cache(&self, nkpts: usize) -> bool {
        let bytes = 16.0 * (self.cell.mol.nao_nr * self.ngrids() * nkpts) as f64;
        bytes < 0.25 * self.max_memory * 1e6
    }

    /// The AO table at `kpts` IF IT IS ALREADY CACHED — never an evaluation.
    ///
    /// [`Fftdf::local_vmat`] needs to know whether the table it is about to
    /// contract exists anyway (an SCF's `get_j`/`get_k` will want it again, so
    /// reducing the cached copy is free) or would have to be built for this
    /// one reduction and then thrown away (where materialising it is pure
    /// cost). `ao_kpts` cannot answer that: it evaluates on a miss.
    ///
    /// The S-08 subset rule is honoured — a band k-list that is a bitwise
    /// subset of the sampling list is served from the sampling table — but
    /// only when that sampling table is itself already cached, since building
    /// it is the evaluation this is trying to avoid.
    fn ao_cached(&self, kpts: &[[f64; 3]]) -> Option<Arc<AoKpts>> {
        let key = kpt_key(kpts);
        if let Ok(c) = self.ao_cache.lock() {
            if let Some(v) = c.get(&key) {
                return Some(Arc::clone(v));
            }
        }
        let reuse = !std::env::var("PYSCF_PBC_BAND_AO_REUSE").is_ok_and(|v| v == "0");
        if !reuse || kpts.len() >= self.kpts.len() {
            return None;
        }
        let full = {
            let c = self.ao_cache.lock().ok()?;
            Arc::clone(c.get(&kpt_key(&self.kpts))?)
        };
        let same =
            |a: &[f64; 3], b: &[f64; 3]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        let map: Vec<usize> = kpts
            .iter()
            .map(|b| self.kpts.iter().position(|k| same(k, b)))
            .collect::<Option<_>>()?;
        Some(Arc::new(AoKpts {
            nao: full.nao,
            ngrids: full.ngrids,
            aot: map.iter().map(|&k| Arc::clone(&full.aot[k])).collect(),
        }))
    }

    /// `v[k][p, q] = Σ_g conj(ao_k[p, g]) vR[g] ao_k[q, g]` — the local half of
    /// `get_nuc` / `get_pp`, and so of `get_hcore`.
    ///
    /// Two routes, bit-identical to each other:
    ///
    /// * the table is already cached — reduce it on the host
    ///   ([`Fftdf::contract_local_potential`]). Nothing is allocated that was
    ///   not going to exist anyway.
    /// * it is not — evaluate and contract in one pass on the device
    ///   ([`pyscf_pbc_gto::eval_ao_kpts_local_vmat`]). The table never crosses
    ///   to the host: of the three live copies of `16 · nkpts · nao · ngrids`
    ///   bytes the first route holds at its peak — the device accumulator, the
    ///   read-back buffer and the per-k planes it keeps — only the accumulator
    ///   remains, and what comes home is the `16 · nkpts · nao²` answer.
    ///
    /// The second route does NOT populate the AO cache, which is both the
    /// point and the reason the default is conditional. Fusing unconditionally
    /// would cost an SCF an extra COLD AO PASS: today `get_hcore` evaluates the
    /// table and leaves it cached for `get_j`/`get_k`, so the run pays one
    /// evaluation; a fused `get_hcore` that kept nothing would make `get_j`
    /// pay a second one. On the cold-AO-dominated cells this tree profiles,
    /// that trade is a loss.
    ///
    /// So [`HcoreFuse::Auto`] — the default — fuses exactly when the table
    /// would NOT have been admitted to the cache
    /// ([`Fftdf::ao_table_fits_cache`]). That predicate already answers "is
    /// this table worth keeping?", and it splits the two cases cleanly:
    ///
    /// * it fits — the old route evaluates once and the whole SCF reuses it;
    ///   fusing would only add an evaluation. Take the host route.
    /// * it does not — the old route evaluates it, reads it back, allocates
    ///   the per-k planes, reduces, and THROWS IT AWAY. Nothing downstream
    ///   benefits and the peak is paid for nothing. Fuse.
    ///
    /// `PYSCF_PBC_HCORE_FUSE=1` forces the fused route and `=0` pins the
    /// pre-K-14f one; those are the two arms the `hcore` profile measures
    /// against each other.
    fn local_vmat(&self, vr: &[f64], kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
        if kpts.is_empty() {
            // The pre-K-14f route sized its output from `kpts.len()`, so an
            // empty k-list yielded no matrices even though `eval_ao_kpts`
            // substitutes gamma internally. Preserved rather than "fixed":
            // every caller passes a real k-list, and redefining what an empty
            // one means is not this change's business.
            return Ok(Vec::new());
        }
        if let Some(ao) = self.ao_cached(kpts) {
            return Ok(self.contract_local_potential(&ao, vr, kpts.len()));
        }
        let fuse = match hcore_fuse_mode() {
            HcoreFuse::Never => false,
            HcoreFuse::Always => true,
            HcoreFuse::Auto => !self.ao_table_fits_cache(kpts.len()),
        };
        if fuse {
            return Ok(pyscf_pbc_gto::eval_ao_kpts_local_vmat(
                &self.cell,
                &self.grids.coords,
                kpts,
                vr,
            )?);
        }
        let ao = self.ao_kpts(kpts)?;
        Ok(self.contract_local_potential(&ao, vr, kpts.len()))
    }

    /// Contract a REAL local potential on the grid into `nao x nao` matrices:
    /// `v[k][p, q] = sum_g conj(ao_k[p, g]) vR[g] ao_k[q, g]`.
    ///
    /// This is the `lib.dot(ao.T.conj() * vR, ao)` of `fft.py:71` and `:110`.
    /// There is NO quadrature weight: `ifft` already carries `1/ngrids` and the
    /// `1/vol` of the inverse Fourier transform cancels the `vol/ngrids` of the
    /// quadrature (see the module docs of `get_nuc`).
    fn contract_local_potential(&self, ao: &AoKpts, vr: &[f64], nkpts: usize) -> Vec<CTensor> {
        let (nao, ngrids) = (ao.nao, ao.ngrids);
        let mut out = Vec::with_capacity(nkpts);
        for k in 0..nkpts {
            let a = &ao.aot[k];
            let mut re = vec![0.0_f64; nao * nao];
            let mut im = vec![0.0_f64; nao * nao];
            for p in 0..nao {
                for q in 0..nao {
                    let mut sr = 0.0_f64;
                    let mut si = 0.0_f64;
                    let (pb, qb) = (p * ngrids, q * ngrids);
                    for g in 0..ngrids {
                        // conj(ao[p,g]) * ao[q,g] * vR[g]
                        let (pr, pi) = (a.re[pb + g], -a.im[pb + g]);
                        let (qr, qi) = (a.re[qb + g], a.im[qb + g]);
                        let w = vr[g];
                        sr += (pr * qr - pi * qi) * w;
                        si += (pr * qi + pi * qr) * w;
                    }
                    re[p * nao + q] = sr;
                    im[p * nao + q] = si;
                }
            }
            out.push(CTensor::from_planes(re, im));
        }
        out
    }
}

/// `vneR` — the nuclear attraction potential in real space (`fft.py:63-67`).
fn nuc_local_potential_r(df: &Fftdf) -> Result<Vec<f64>, PbcDfError> {
    let cell = &df.cell;
    let mesh = df.mesh;
    let gv = get_gv(cell, Some(mesh))?;
    let ngrids = gv.len();
    // fft.py:63 — `cell.get_SI(mesh=mesh)` passes no Gv, i.e. the SEPARABLE
    // branch (products of per-axis phases with numpy's FMA complex multiply),
    // not the dense `exp(-1j*coords@Gv.T)`. He at the origin hid this: every
    // phase there is exactly 1 either way.
    let si = get_si(cell, None, Some(mesh), None)?;
    let charges = cell.atom_charges();
    let natm = cell.mol.natm;

    // fft.py:64 — `rhoG = numpy.dot(charge, SI)`: plain left-to-right
    // accumulation over atoms, no FMA (natm up to 64 probed).
    let mut rho_re = vec![0.0_f64; ngrids];
    let mut rho_im = vec![0.0_f64; ngrids];
    for ia in 0..natm {
        let z = -(charges[ia] as f64);
        let base = ia * ngrids;
        for g in 0..ngrids {
            rho_re[g] += z * si.re[base + g];
            rho_im[g] += z * si.im[base + g];
        }
    }

    // fft.py:65-67
    let coulg = get_coulg_at_gv(cell, mesh, &gv)?;
    for g in 0..ngrids {
        rho_re[g] *= coulg[g];
        rho_im[g] *= coulg[g];
    }
    let vneg = CTensor::from_planes(rho_re, rho_im);
    let vner = ifft_upstream(&vneg, mesh)?.re;
    Ok(vner)
}

/// `KNumInt.block_loop` (`pyscf/pbc/dft/numint.py:1058-1100`) as seen from
/// `FFTDF.aoR_loop` (`pyscf/pbc/df/fft.py:297-319`) for `get_nuc` (`deriv = 0`,
/// hence `comp = 1`).
///
/// Upstream computes, with `BLKSIZE = 56`:
///
/// ```text
/// blksize = int(max_memory*1e6/(comp*2*nao*16*BLKSIZE))
/// blksize = max(4, min(blksize, ngrids//BLKSIZE+1, 2400)) * BLKSIZE
/// ```
///
/// `nkpts` is deliberately NOT in the formula — `block_loop` sizes the grid
/// blocks from `comp`, `nao` and `max_memory` only (checked against the source
/// line before writing this; the parameter is kept so call sites read like the
/// Python call).
///
/// `max_memory_mb` is the constant `2000.0`: Rust has no `lib.current_memory()`
/// RSS reading, and the oracle pins `mydf.max_memory = 0`, so upstream's
/// `max(2000, 0 - rss)` is exactly `2000`. Do NOT pass `df.max_memory` here.
pub fn aor_loop_blocks(
    ngrids: usize,
    nao: usize,
    _nkpts: usize,
    max_memory_mb: f64,
) -> Vec<(usize, usize)> {
    const BLKSIZE: usize = 56;
    const COMP: f64 = 1.0; // deriv = 0
    let blksize =
        (max_memory_mb * 1e6 / (COMP * 2.0 * nao as f64 * 16.0 * BLKSIZE as f64)) as usize;
    let blksize = (4usize.max(blksize.min(ngrids / BLKSIZE + 1).min(2400))) * BLKSIZE;
    let mut out = Vec::new();
    let mut p0 = 0;
    while p0 < ngrids {
        let p1 = (p0 + blksize).min(ngrids);
        out.push((p0, p1));
        p0 = p1;
    }
    out
}
/// `get_nuc(mydf, kpts)` — `fft.py:40-80`.
///
/// ```text
/// rhoG  = -sum_a Z_a SI[a, G]          (nuclear charge density in G space)
/// vneG  = rhoG * coulG
/// vneR  = ifft(vneG).real
/// vne_k = sum_g conj(ao_k) vneR ao_k
/// ```
///
/// # Why there is no `vol/ngrids` factor
///
/// The real-space potential is `V(r) = (1/vol) sum_G vneG e^{iGr}` while `ifft`
/// computes `(1/ngrids) sum_G ...`, and the quadrature that follows carries
/// `vol/ngrids`. The two cancel exactly, which is why upstream's line 71 has no
/// weight in it. Adding one is the classic factor-of-`vol` bug here.
///
/// # Upstream's bits
///
/// Every stage follows upstream's rounding, not just its mathematics, so the
/// result is bit-identical to PySCF 2.12.1's `FFTDF.get_nuc` run with
/// `OMP_NUM_THREADS=1` (He/STO-3G 2x2x2, mesh 11 — `tests/fftdf.rs`):
///
/// * `vneR` through [`pyscf_pbc_tools::ifft_upstream`] — the pocketfft route
///   (`cfftp`/Bluestein per axis) or, on all-`_EXCLUDE` meshes, upstream's
///   `_ifftn_blas` GEMM route;
/// * the AO table through [`pyscf_pbc_gto::eval_ao_kpts_upstream`]
///   (`PBCeval_sph_iter`'s screen and image sum), not the cached
///   [`Fftdf::ao_kpts`];
/// * `lib.dot(ao.T.conj()*vneR, ao)` through the Barcelona OpenBLAS emulation
///   ([`pyscf_algebra::openblas_emu`]).
///
/// Where a stage is out of reach — a basis with `l >= 5`, or more than one
/// atom (`numpy.dot(charge, SI)` runs on numpy's own BLAS, unmodelled) — that
/// stage keeps this port's own arithmetic, which agrees to ~1e-12 but not to
/// the bit. Upstream's multi-threaded `NPdgemm` merges per-thread partial sums
/// in `omp critical` order, so its own bits vary with the thread count; the
/// single-threaded result is the reference.
///
/// # Errors
/// Propagates the G-vector, structure-factor, `coulG` and AO evaluations.
pub fn get_nuc(df: &Fftdf, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
    let cell = &df.cell;
    let vner = nuc_local_potential_r(df)?;

    match eval_ao_kpts_upstream(cell, &df.grids.coords, kpts)? {
        Some(ao) => Ok(contract_local_potential_upstream(&ao, &vner)),
        None => df.local_vmat(&vner, kpts),
    }
}

/// `vne[k] = 0 + lib.dot(ao.T.conj() * vR, ao)` (`fft.py:70-74`) in upstream's
/// rounding, over every `aoR_loop` block.
///
/// `lib.dot` lands in `dgemm_`/`zgemm_('N', 'T', nao, nao, kblk, ao, w)` with
/// `w = conj(ao) * vR` (`NPdgemm` swaps the operands to get a row-major
/// result), so `C[i + j*nao] = sum_g ao_i(g) w_j(g)` is `vne[j, i]` — the
/// column-major `C` IS the row-major `vne`. A gamma k-point's AO table is real
/// upstream, so it takes the real `dgemm_` and has an exactly-zero imaginary
/// part.
///
/// Each block's GEMM result passes through `c = 0; c += cpriv` (`NPdgemm`'s
/// `+ 0.0` normalisation) and then accumulates `vne[k] += block` element-wise
/// from `0.0`. When `NPdgemm`'s `if ((k/m) > 3 && (k/n) > 3)` branch is false
/// upstream calls `dgemm_` directly with `beta = 0` (no `+ 0.0` step), but
/// OpenBLAS zeroes `C` first so the value is identical — the `+ 0.0` is kept.
/// The AO table stays whole-grid: `blksize` is a multiple of the 56-point
/// `BLKSIZE`, so per-block evaluation is identical point by point.
fn contract_local_potential_upstream(
    ao: &pyscf_pbc_gto::EvalAoKptsOutput,
    vr: &[f64],
) -> Vec<CTensor> {
    let (nao, ngrids) = (ao.nao, ao.ngrids);
    let blocks = aor_loop_blocks(ngrids, nao, ao.kaos.len(), 2000.0);
    ao.kaos
        .iter()
        .zip(&ao.gamma)
        .map(|(a, &gamma)| {
            let mut acc_re = vec![0.0; nao * nao];
            let mut acc_im = vec![0.0; nao * nao];
            for &(p0, p1) in &blocks {
                let kblk = p1 - p0;
                // Column-major `nao x kblk` operands over this block's columns:
                // A[i + g*nao] = ao_i(p0+g).
                let mut a_re = vec![0.0; nao * kblk];
                let mut w_re = vec![0.0; nao * kblk];
                for i in 0..nao {
                    for (gg, g) in (p0..p1).enumerate() {
                        let x = a.re[i * ngrids + g];
                        a_re[i + gg * nao] = x;
                        w_re[i + gg * nao] = x * vr[g];
                    }
                }
                let mut c_re = vec![0.0; nao * nao];
                let mut c_im = vec![0.0; nao * nao];
                if gamma {
                    openblas_emu::dgemm_nt(nao, nao, kblk, &a_re, &w_re, &mut c_re);
                } else {
                    let mut a_im = vec![0.0; nao * kblk];
                    let mut w_im = vec![0.0; nao * kblk];
                    for i in 0..nao {
                        for (gg, g) in (p0..p1).enumerate() {
                            let y = a.im[i * ngrids + g];
                            a_im[i + gg * nao] = y;
                            // conj(ao) * vR: numpy's complex-by-real product.
                            w_im[i + gg * nao] = -y * vr[g];
                        }
                    }
                    openblas_emu::zgemm_nt(
                        nao, nao, kblk, &a_re, &a_im, &w_re, &w_im, &mut c_re, &mut c_im,
                    );
                }
                for v in c_re.iter_mut().chain(c_im.iter_mut()) {
                    *v += 0.0;
                }
                for i in 0..nao * nao {
                    acc_re[i] += c_re[i];
                    acc_im[i] += c_im[i];
                }
            }
            CTensor::from_planes(acc_re, acc_im)
        })
        .collect()
}

/// `get_pp(mydf, kpts)` — `fft.py:82-178`, with the non-local half taken from
/// Phase 10's real-space route (see the module docs).
///
/// # Errors
/// Propagates the G-space local factors, the FFT, `get_pp_nl` and the AO
/// evaluation.
pub fn get_pp(df: &Fftdf, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
    let cell = &df.cell;
    let vpplocr = pp_local_potential_r(df)?;
    let mut vpp = df.local_vmat(&vpplocr, kpts)?;
    vpp_add_nonlocal(cell, kpts, &mut vpp)?;
    Ok(vpp)
}

/// `vpplocR` — the local GTH potential in real space (`fft.py:101-112`).
fn pp_local_potential_r(df: &Fftdf) -> Result<Vec<f64>, PbcDfError> {
    let cell = &df.cell;
    let mesh = df.mesh;
    let gv = get_gv(cell, Some(mesh))?;
    let ngrids = gv.len();
    let si = get_si(cell, Some(&gv), None, None)?;
    let natm = cell.mol.natm;

    // fft.py:101-103 — vpplocG = -einsum('ij,ij->j', SI, get_vlocG(cell, Gv)).
    let vlocg = pyscf_pbc_gto::pseudo::get_vlocg(cell, &gv)?;
    let mut re = vec![0.0_f64; ngrids];
    let mut im = vec![0.0_f64; ngrids];
    for ia in 0..natm {
        let base = ia * ngrids;
        for g in 0..ngrids {
            re[g] -= si.re[base + g] * vlocg[base + g];
            im[g] -= si.im[base + g] * vlocg[base + g];
        }
    }

    // fft.py:106-112 — the local part, evaluated in real space.
    Ok(ifft(&CTensor::from_planes(re, im), mesh)?.re)
}

/// `vpp += V_nl` and the gamma-point `.real` of `fft.py:114-176`.
fn vpp_add_nonlocal(
    cell: &Cell,
    kpts: &[[f64; 3]],
    vpp: &mut [CTensor],
) -> Result<(), PbcDfError> {
    // fft.py:114-176 — the non-local part. Phase 10 owns it in real space.
    let vnl = pyscf_pbc_gto::pseudo::get_pp_nl(cell, kpts)?;
    let nao = cell.mol.nao_nr;
    for (k, v) in vpp.iter_mut().enumerate() {
        // Phase-10 output is F-order (see `zlinalg::forder_to_c`).
        let nl = forder_to_c(&vnl[k], nao, nao);
        zadd_assign(v, &nl);
        // fft.py:172-175 — a gamma-point block is real by construction.
        if pyscf_pbc_gto::is_zero(&kpts[k]) {
            for t in v.im.iter_mut() {
                *t = 0.0;
            }
        }
    }
    Ok(())
}

/// BAND-03 — `hcore` at `kpts` WITHOUT its local grid term: `T + V_nl` for a
/// pseudopotential cell, `T` for an all-electron one. Adding
/// `Σ_g conj(ao_p) ao_q v[g]` with `v` from [`PeriodicDf::local_potential_r`]
/// gives [`get_hcore`] back (up to summation order).
///
/// # Errors
/// Propagates `get_pp_nl` and `pbc_intor('int1e_kin')`.
pub fn get_hcore_nonlocal(cell: &Cell, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
    let nao = cell.mol.nao_nr;
    let mut h = vec![CTensor::zeros(nao * nao); kpts.len()];
    if cell.pseudo.is_some() {
        vpp_add_nonlocal(cell, kpts, &mut h)?;
    }
    let t = pyscf_pbc_gto::get_t(cell, kpts)?;
    for (k, m) in h.iter_mut().enumerate() {
        zadd_assign(m, &forder_to_c(&t[k], nao, nao));
    }
    Ok(h)
}

/// `get_hcore` for a periodic cell — `khf.py:66-90`.
///
/// `T + V_pp` for a pseudopotential cell, `T + V_ne` for an all-electron one.
/// This is the function `pyscf_pbc_gto::hcore::get_hcore` deferred to Phase 11.
///
/// # Errors
/// Propagates [`get_pp`] / [`get_nuc`] and `pbc_intor('int1e_kin')`.
///
/// Takes `&dyn PeriodicDf` since plan 13-07 (D-PBC-22): the body is builder
/// agnostic — it picks `get_pp` vs `get_nuc` from `cell.pseudo` and adds
/// `int1e_kin` — so binding it to `Fftdf` was the only thing stopping a driver
/// from running on AFTDF.
pub fn get_hcore(df: &dyn PeriodicDf, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
    let cell = df.cell();
    let nao = cell.mol.nao_nr;
    let mut nuc = if cell.pseudo.is_some() {
        df.get_pp(kpts)?
    } else {
        df.get_nuc(kpts)?
    };
    let t = pyscf_pbc_gto::get_t(cell, kpts)?;
    for (k, h) in nuc.iter_mut().enumerate() {
        zadd_assign(h, &forder_to_c(&t[k], nao, nao));
    }
    Ok(nuc)
}

impl PeriodicDf for Fftdf {
    fn cell(&self) -> &Cell {
        &self.cell
    }
    fn mesh(&self) -> [usize; 3] {
        self.mesh
    }
    fn name(&self) -> &'static str {
        "FFTDF"
    }
    fn kpts(&self) -> &[[f64; 3]] {
        &self.kpts
    }
    fn build(&mut self) -> Result<(), PbcDfError> {
        let kpts = self.kpts.clone();
        self.ao_kpts(&kpts)?;
        Ok(())
    }
    fn get_nuc(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
        get_nuc(self, kpts)
    }
    fn get_pp(&self, kpts: &[[f64; 3]]) -> Result<Vec<CTensor>, PbcDfError> {
        get_pp(self, kpts)
    }
    fn local_potential_r(&self) -> Result<Option<Vec<f64>>, PbcDfError> {
        if self.cell.pseudo.is_some() {
            pp_local_potential_r(self).map(Some)
        } else {
            nuc_local_potential_r(self).map(Some)
        }
    }
    fn get_jk(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        opts: JkOpts<'_>,
    ) -> Result<JkResult, PbcDfError> {
        let vj = if opts.with_j {
            Some(crate::fft_jk::get_j_kpts(
                self,
                dms,
                opts.hermi,
                kpts,
                opts.kpts_band,
                opts.omega,
            )?)
        } else {
            None
        };
        let vk = if opts.with_k {
            Some(crate::fft_jk::get_k_kpts_opts(
                self,
                dms,
                opts.hermi,
                kpts,
                opts.kpts_band,
                opts.exxdiv,
                opts.omega,
                opts.kk_symmetry,
            )?)
        } else {
            None
        };
        Ok(JkResult { vj, vk })
    }

    /// `FFTDF.get_jk_e1` — the only density-fitting route in PySCF 2.12.1
    /// that has a gradient (`fft.py:324-328`). Refuses a carried k-pair flag
    /// (clause 4b); see the inherent method in `crate::fft_jk_grad`.
    fn get_jk_e1(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        opts: JkOpts<'_>,
        mo: Option<&crate::fft_jk_grad::TaggedMo>,
    ) -> Result<crate::fft_jk_grad::GradJkResult, PbcDfError> {
        self.get_jk_e1(dms, kpts, opts, mo)
    }

    /// `FFTDF.get_j_e1` (`fft.py:330-333`).
    fn get_j_e1(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        kpts_band: Option<&[[f64; 3]]>,
    ) -> Result<crate::fft_jk_grad::GradMats, PbcDfError> {
        self.get_j_e1(dms, kpts, kpts_band)
    }

    /// `FFTDF.get_k_e1` (`fft.py:335-340`).
    fn get_k_e1(
        &self,
        dms: &[KMats],
        kpts: &[[f64; 3]],
        kpts_band: Option<&[[f64; 3]]>,
        exxdiv: Option<ExxDiv>,
        omega: Option<f64>,
        mo: Option<&crate::fft_jk_grad::TaggedMo>,
    ) -> Result<crate::fft_jk_grad::GradMats, PbcDfError> {
        self.get_k_e1(dms, kpts, kpts_band, exxdiv, omega, mo)
    }

    fn ao2mo(
        &self,
        mos: [&crate::MoCoeff; 4],
        kidx: [usize; 4],
        _compact: bool,
    ) -> Result<crate::Eri, PbcDfError> {
        let k = kidx.map(|i| self.kpts[i]);
        crate::pbc_ao2mo::fft_general_mo_first(self, mos, k, None)
    }

    fn ao2mo_cached(
        &self,
        mos: [&crate::MoCoeff; 4],
        kidx: [usize; 4],
        _compact: bool,
        cache: Option<&crate::CoulGCache>,
    ) -> Result<crate::Eri, PbcDfError> {
        let k = kidx.map(|i| self.kpts[i]);
        crate::pbc_ao2mo::fft_general_mo_first(self, mos, k, cache)
    }

    fn get_ao_eri(&self, kidx: [usize; 4], _compact: bool) -> Result<crate::Eri, PbcDfError> {
        let k = kidx.map(|i| self.kpts[i]);
        let data = crate::pbc_ao2mo::fft_get_eri(self, k)?;
        let d = crate::PairDims::plain(self.cell.mol.nao_nr, self.cell.mol.nao_nr);
        Ok(crate::Eri {
            data,
            row: d,
            col: d,
        })
    }

    fn ao2mo_7d(&self, mos: crate::MoKpts<'_>, factor: f64) -> Result<crate::Eri7d, PbcDfError> {
        crate::pbc_ao2mo::fft_ao2mo_7d(self, mos, factor)
    }
}

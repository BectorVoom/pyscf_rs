//! `kpts_helper` — port of `pyscf/pbc/lib/kpts_helper.py`.
//!
//! Plan 09-06 seeded this module with `round_to_fbz` (and the `lib.cleanse` it
//! calls), which `pyscf_pbc_tools::lattice::round_to_cell0` needs. Plan 09-07
//! added `is_zero`, `member`, `intersection`, `unique`, `get_kconserv` and
//! `get_kconserv3`.
//!
//! # Why nothing here takes a `Cell`
//!
//! `pyscf-pbc-lib` is the BOTTOM of the periodic DAG (PBC-MASTER-PLAN §4:
//! `pyscf-pbc-lib -> pyscf-pbc-tools -> pyscf-pbc-gto`), so it cannot name
//! `Cell`. Upstream's `get_kconserv(cell, kpts)` / `get_kconserv3` read exactly
//! one thing off the cell — `cell.lattice_vectors()` — so the ports take that
//! `a` directly and `pyscf_pbc_gto::kpts_mesh` holds the `Cell`-taking
//! wrappers. Same split plan 09-04 and 09-06 used.
//!
//! # Still missing (later plans)
//!
//! `unique_with_wrap_around`, `members_with_wrap_around`,
//! `group_by_conj_pairs`, `conj_mapping`, `kk_adapted_iter`, `KptsHelper` and
//! `get_kconserv_ria` — Phase 13/16/17 consumers, not Phase 9.

use std::cmp::Ordering;

/// `kpts_helper.py:31` — two k-points differing by less than this are the same
/// point. **This exact threshold decides real-vs-complex code paths
/// everywhere. Do not change it.**
pub const KPT_DIFF_TOL: f64 = 1e-6;

/// `np.mod(x, 1)` — the Python/NumPy modulo, whose result carries the sign of
/// the (positive) divisor, unlike C's `fmod` / Rust's `%`.
fn mod1(x: f64) -> f64 {
    x - x.floor()
}

/// `lib.cleanse(a, axis=0, tol)` for an `(n, 3)` array — port of
/// `pyscf/lib/numpy_helper.py:1561-1602`.
///
/// `axis = 0` means the comparison runs along the FIRST axis, i.e. within each
/// of the three coordinate columns independently. Values are sorted, split into
/// clusters wherever a consecutive difference exceeds `tol`, and every member of
/// a cluster is overwritten with the cluster's SMALLEST value — which is what
/// lets a subsequent `round` + `unique` behave as intended.
fn cleanse_columns(v: &mut [[f64; 3]], tol: f64) {
    let n = v.len();
    #[allow(clippy::needless_range_loop)] // `j` indexes the INNER [f64; 3], not `v`.
    for j in 0..3 {
        // `numpy.argsort` — ties are irrelevant here because equal values are
        // mapped to the same output value either way.
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&x, &y| v[x][j].partial_cmp(&v[y][j]).unwrap_or(Ordering::Equal));
        // Snapshot the sorted values BEFORE writing, so a cluster boundary is
        // decided on the original data (upstream reads `sorted_a_flat`, which
        // is a copy, while writing into `a_flat`).
        let sorted: Vec<f64> = idx.iter().map(|&p| v[p][j]).collect();
        let mut i = 0;
        while i < n {
            let mut k = i + 1;
            while k < n && sorted[k] - sorted[k - 1] <= tol {
                k += 1;
            }
            let first = sorted[i];
            for &p in &idx[i..k] {
                v[p][j] = first;
            }
            i = k;
        }
    }
}

/// Round scaled k-points into the first Brillouin zone.
/// Ports `kpts_helper.py:65-88`.
///
/// `wrap_around = false` folds to `[0, 1)`; `true` folds to `[-0.5, 0.5)`.
/// `tol` sets both the `cleanse` clustering width and the rounding precision
/// (`decimal = -int(log10((tol + 1e-16) / 10))`).
pub fn round_to_fbz(kpts: &[[f64; 3]], wrap_around: bool, tol: f64) -> Vec<[f64; 3]> {
    // decimal = -np.log10((tol+1e-16)/10.).astype(int)  — `astype(int)` binds
    // to the log, and truncates TOWARD ZERO, so tol = 1e-6 gives decimal = 6.
    let decimal = -(((tol + 1e-16) / 10.0).log10().trunc() as i32);
    let scale = 10.0_f64.powi(decimal);

    // kpts_fbz = np.mod(kpts, 1)
    let mut fbz: Vec<[f64; 3]> = kpts
        .iter()
        .map(|k| [mod1(k[0]), mod1(k[1]), mod1(k[2])])
        .collect();
    // kpts_fbz = lib.cleanse(kpts_fbz, axis=0, tol=tol)
    cleanse_columns(&mut fbz, tol);
    for k in fbz.iter_mut() {
        for x in k.iter_mut() {
            // kpts_fbz.round(decimal) — NumPy rounds halves to EVEN.
            *x = (*x * scale).round_ties_even() / scale;
            // kpts_fbz = np.mod(kpts_fbz, 1)
            *x = mod1(*x);
        }
    }
    if wrap_around {
        for k in fbz.iter_mut() {
            for x in k.iter_mut() {
                if *x >= 0.5 {
                    *x -= 1.0;
                }
            }
        }
    }
    fbz
}

/// `is_zero(kpt)` — `kpts_helper.py:31-32`. `sum |k_i| < KPT_DIFF_TOL`.
///
/// **This exact predicate decides the real-vs-complex code path everywhere.**
/// Note the threshold is [`KPT_DIFF_TOL`] = 1e-6, NOT 1e-9: PBC-MASTER-PLAN
/// §8.1 plan 09-07 step 2 quotes 1e-9, but `kpts_helper.py:32` reads
/// `abs(np.asarray(kpt)).sum() < KPT_DIFF_TOL` and RULE 2 makes the Python
/// authoritative. Upstream behaviour confirms it: `is_zero([1e-7, 0, 0])` is
/// true and `is_zero([1e-6, 0, 0])` is false.
pub fn is_zero(kpt: &[f64]) -> bool {
    kpt.iter().map(|x| x.abs()).sum::<f64>() < KPT_DIFF_TOL
}

/// `is_gamma_point` — `kpts_helper.py:37`, an alias of [`is_zero`].
pub fn is_gamma_point(kpt: &[f64]) -> bool {
    is_zero(kpt)
}

/// `gamma_point` — `kpts_helper.py:37`, an alias of [`is_zero`].
pub fn gamma_point(kpt: &[f64]) -> bool {
    is_zero(kpt)
}

/// `is_trim(cell, kpts, tol)` — `kpts_helper.py:39-63`. Whether each k-point
/// is a time-reversal-invariant momentum (TRIM), i.e. `k == -k mod G`.
///
/// `khf_ksymm.py:126` needs this for the `eig_trs` branch. It was the one
/// `kpts_helper` function still missing from this module (17-CONTEXT §5).
///
/// Takes `a = cell.lattice_vectors()` rather than a `Cell` — this crate is
/// the BOTTOM of the periodic DAG and cannot name `Cell`; the scaled-k-point
/// conversion `cell.get_scaled_kpts` performs is `abs . a.T / (2*pi)` and
/// nothing else. `pyscf_pbc_gto::kpts_mesh::is_trim` is the `Cell`-taking
/// wrapper, the same split `get_kconserv` already uses.
///
/// The rounding is upstream's, not a tolerance comparison:
/// `logtol = ceil(-log10(tol))`, then `round(2*k_scaled, logtol+1) % 1`, and
/// the point is a TRIM when the largest component of THAT is below `tol`.
pub fn is_trim(a: &[[f64; 3]; 3], kpts: &[[f64; 3]], tol: f64) -> Vec<bool> {
    // logtol = np.ceil(-np.log10(tol)).astype(int)
    let logtol = (-tol.log10()).ceil() as i32;
    let scale = 10.0_f64.powi(logtol + 1);
    // scaled_kpts = cell.get_scaled_kpts(kpts) == kpts . a.T / (2*pi)
    let inv_2pi = 1.0 / (2.0 * std::f64::consts::PI);
    kpts.iter()
        .map(|k| {
            let mut worst = 0.0_f64;
            for row in a.iter() {
                let ks = (row[0] * k[0] + row[1] * k[1] + row[2] * k[2]) * inv_2pi;
                // np.round(2*scaled, logtol+1) % 1 — NumPy rounds halves to
                // even, and its `%` carries the sign of the divisor.
                let r = (2.0 * ks * scale).round_ties_even() / scale;
                let m = r - r.floor();
                worst = worst.max(m);
            }
            worst < tol
        })
        .collect()
}

/// `member(kpt, kpts)` — `kpts_helper.py:90-97`. The ASCENDING indices of the
/// rows of `kpts` whose Chebyshev distance to `kpt` is below [`KPT_DIFF_TOL`].
pub fn member(kpt: &[f64; 3], kpts: &[[f64; 3]]) -> Vec<usize> {
    kpts.iter()
        .enumerate()
        .filter(|(_, k)| chebyshev(k, kpt) < KPT_DIFF_TOL)
        .map(|(i, _)| i)
        .collect()
}

/// `intersection(kpts1, kpts2)` — `kpts_helper.py:99-106`. The ASCENDING,
/// de-duplicated indices INTO `kpts1` of rows that also occur in `kpts2`.
pub fn intersection(kpts1: &[[f64; 3]], kpts2: &[[f64; 3]]) -> Vec<usize> {
    kpts1
        .iter()
        .enumerate()
        .filter(|(_, k1)| kpts2.iter().any(|k2| chebyshev(k1, k2) < KPT_DIFF_TOL))
        .map(|(i, _)| i)
        .collect()
}

/// `abs(a - b).max()` — the Chebyshev distance upstream compares against
/// [`KPT_DIFF_TOL`] in `member` / `intersection`.
fn chebyshev(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    (a[0] - b[0])
        .abs()
        .max((a[1] - b[1]).abs())
        .max((a[2] - b[2]).abs())
}

/// The return of [`unique`] — upstream's
/// `(uniq_kpts, uniq_index, uniq_inverse)` triple.
#[derive(Debug, Clone, PartialEq)]
pub struct UniqueKpts {
    /// The unique k-points, in order of FIRST OCCURRENCE in the input.
    pub kpts: Vec<[f64; 3]>,
    /// `index[u]` — the input row that first produced `kpts[u]`.
    pub index: Vec<usize>,
    /// `inverse[i]` — which entry of `kpts` input row `i` maps to.
    pub inverse: Vec<usize>,
}

/// `unique(kpts)` — `kpts_helper.py:108-142`.
///
/// Upstream's docstring says "sorted", but the `argsort`/`argsort` pair on
/// `uniq_index` (`:122-125`) undoes NumPy's lexicographic order, so the result
/// is in order of FIRST OCCURRENCE. This port reproduces that directly.
///
/// Two k-points are the same when their coordinates AGREE AFTER ROUNDING to
/// `digits = int(-log10(KPT_DIFF_TOL)) = 6` decimals — not when they are within
/// `KPT_DIFF_TOL` of each other. That distinction matters at a rounding
/// boundary (0.4999995 and 0.5000005 both round to 0.5 and merge, while
/// 0.4999994 and 0.5000004 do not), and it is upstream's behaviour.
pub fn unique(kpts: &[[f64; 3]]) -> UniqueKpts {
    // digits = int(-np.log10(KPT_DIFF_TOL))
    let digits = -KPT_DIFF_TOL.log10().trunc() as i32;
    let scale = 10.0_f64.powi(digits);
    // np.unique(kpts.round(digits), axis=0) compares the ROUNDED rows exactly.
    let key = |k: &[f64; 3]| -> [u64; 3] {
        let mut out = [0_u64; 3];
        for (o, x) in out.iter_mut().zip(k.iter()) {
            let r = (x * scale).round_ties_even() / scale;
            // NumPy compares by value, so -0.0 and 0.0 are the same key.
            *o = (if r == 0.0 { 0.0 } else { r }).to_bits();
        }
        out
    };

    let mut seen: Vec<[u64; 3]> = Vec::new();
    let mut out = UniqueKpts {
        kpts: Vec::new(),
        index: Vec::new(),
        inverse: vec![0; kpts.len()],
    };
    for (i, k) in kpts.iter().enumerate() {
        let ki = key(k);
        match seen.iter().position(|s| *s == ki) {
            Some(u) => out.inverse[i] = u,
            None => {
                seen.push(ki);
                out.kpts.push(*k);
                out.index.push(i);
                out.inverse[i] = seen.len() - 1;
            }
        }
    }
    out
}

/// The tolerance `get_kconserv` / `get_kconserv3` use on the summed fractional
/// error of `(k_K - k_L + k_M - k_N) . a / (2*pi)` (`kpts_helper.py:323`,
/// `:433`). It is NOT [`KPT_DIFF_TOL`].
pub const KCONSERV_TOL: f64 = 1e-9;

/// Momentum-conservation table — upstream's `kconserv[K, L, M] = N`.
///
/// The plan specifies "`Vec<i32>` shaped `[nk][nk][nk]`"; [`Kconserv::data`] is
/// exactly that flat C-order vector, and [`Kconserv::get`] is the index
/// arithmetic so no caller has to re-derive it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kconserv {
    /// Number of k-points along each of the three axes of the table.
    pub nkpts: usize,
    /// `nkpts^3` entries, C-order over `(K, L, M)`.
    pub data: Vec<i32>,
}

impl Kconserv {
    /// `kconserv[k, l, m]` — the index `n` with `(k_k - k_l + k_m - k_n) . a`
    /// an integer multiple of `2*pi`.
    ///
    /// # Panics
    /// If any index is `>= nkpts`.
    pub fn get(&self, k: usize, l: usize, m: usize) -> i32 {
        assert!(k < self.nkpts && l < self.nkpts && m < self.nkpts);
        self.data[(k * self.nkpts + l) * self.nkpts + m]
    }
}

/// `w`-th component of `v . a.T / (2*pi)`: `sum_x a[w][x] * v[x]`, with
/// `a` ALREADY divided by `2*pi` (`kpts_helper.py:315`).
fn frac_component(a_over_2pi: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    let mut s = [0.0_f64; 3];
    for (w, sw) in s.iter_mut().enumerate() {
        *sw = a_over_2pi[w][0] * v[0] + a_over_2pi[w][1] * v[1] + a_over_2pi[w][2] * v[2];
    }
    s
}

/// `sum_w |s_w - rint(s_w)| < KCONSERV_TOL` — upstream's integrality test.
fn is_integral(s: &[f64; 3]) -> bool {
    s.iter()
        .map(|x| (x - x.round_ties_even()).abs())
        .sum::<f64>()
        < KCONSERV_TOL
}

/// `get_kconserv(cell, kpts)` — `kpts_helper.py:291-325`.
///
/// `kconserv[K, L, M]` is the index `N` satisfying
/// `(k_K - k_L + k_M - k_N) . a = 2*n*pi`.
///
/// `a` is `cell.lattice_vectors()`; the `/(2*pi)` of `:315` happens inside.
///
/// **Deviation:** upstream first tries a `k2gamma`-based shortcut
/// (`kpts_to_kmesh` + `double_translation_indices`, `:303-311`) and falls back
/// to `_get_kconserv_slow` (`:313-325`) when the k-points are not a full
/// Monkhorst-Pack mesh. This port implements `_get_kconserv_slow` ONLY.
/// `pyscf/pbc/tools/k2gamma.py` is not in this plan's PORT block, and the two
/// paths were verified to produce IDENTICAL tables on every probed mesh
/// (`[1,1,1]`, `[2,2,1]`, `[2,2,2]`, `[3,1,2]`, `[3,3,3]`, `[4,2,1]`, and the
/// `with_gamma_point=False` / `wrap_around=True` / `scaled_center` variants).
/// The shortcut is a performance optimisation; see the plan SUMMARY.
pub fn get_kconserv(a: &[[f64; 3]; 3], kpts: &[[f64; 3]]) -> Kconserv {
    let nkpts = kpts.len();
    // a = cell.lattice_vectors() / (2*np.pi)
    let mut a2 = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            a2[i][j] = a[i][j] / (2.0 * std::f64::consts::PI);
        }
    }
    let mut kconserv = vec![0_i32; nkpts * nkpts * nkpts];
    // kvKLM = kpts[:,None,None,:] - kpts[:,None,:] + kpts
    for k in 0..nkpts {
        for l in 0..nkpts {
            for m in 0..nkpts {
                let kv = [
                    kpts[k][0] - kpts[l][0] + kpts[m][0],
                    kpts[k][1] - kpts[l][1] + kpts[m][1],
                    kpts[k][2] - kpts[l][2] + kpts[m][2],
                ];
                // for N, kvN in enumerate(kpts): ... kconserv[mask] = N
                // A later N overwrites an earlier one, exactly as upstream.
                for (n, kvn) in kpts.iter().enumerate() {
                    let d = [kv[0] - kvn[0], kv[1] - kvn[1], kv[2] - kvn[2]];
                    if is_integral(&frac_component(&a2, &d)) {
                        kconserv[(k * nkpts + l) * nkpts + m] = n as i32;
                    }
                }
            }
        }
    }
    Kconserv {
        nkpts,
        data: kconserv,
    }
}

/// One entry of `get_kconserv3`'s `kijkab` argument: either a whole index list
/// or a single pinned k-point index. A pinned axis is DROPPED from the output
/// shape (`kpts_helper.py:436-438`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KIdx {
    /// `kijkab[x]` given as a plain integer.
    One(usize),
    /// `kijkab[x]` given as an index array.
    Many(Vec<usize>),
}

impl KIdx {
    /// The indices this entry selects — length 1 for [`KIdx::One`].
    pub fn indices(&self) -> &[usize] {
        match self {
            KIdx::One(i) => std::slice::from_ref(i),
            KIdx::Many(v) => v,
        }
    }
    /// `np.size(x)` — the axis length BEFORE the pinned-axis squeeze.
    pub fn len(&self) -> usize {
        self.indices().len()
    }
    /// Whether this axis selects nothing.
    pub fn is_empty(&self) -> bool {
        self.indices().is_empty()
    }
    /// Whether upstream would drop this axis from the output shape.
    pub fn is_scalar(&self) -> bool {
        matches!(self, KIdx::One(_))
    }
}

/// The return of [`get_kconserv3`] — a dense integer array of rank
/// `shape.len()` (`<= 5`, one axis per non-scalar `kijkab` entry).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Kconserv3 {
    /// Output shape after upstream's pinned-axis squeeze (`:436-438`).
    pub shape: Vec<usize>,
    /// `shape.iter().product()` entries, C-order.
    pub data: Vec<i32>,
}

/// `get_kconserv3(cell, kpts, kijkab)` — `kpts_helper.py:409-439`.
///
/// Finds the `kc` with `(ki + kj + kk - ka - kb - kc) . a = 2*n*pi`, for every
/// `(ki, kj, kk, ka, kb)` the five `kijkab` entries select.
///
/// `a` is `cell.lattice_vectors()`.
///
/// # Panics
/// If `kijkab` does not have exactly 5 entries, or an index is out of range.
pub fn get_kconserv3(a: &[[f64; 3]; 3], kpts: &[[f64; 3]], kijkab: &[KIdx]) -> Kconserv3 {
    assert_eq!(
        kijkab.len(),
        5,
        "kijkab must have 5 entries (ki, kj, kk, ka, kb)"
    );
    let mut a2 = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            a2[i][j] = a[i][j] / (2.0 * std::f64::consts::PI);
        }
    }
    let pick = |x: &KIdx| -> Vec<[f64; 3]> { x.indices().iter().map(|&i| kpts[i]).collect() };
    let (ki, kj, kk, ka, kb) = (
        pick(&kijkab[0]),
        pick(&kijkab[1]),
        pick(&kijkab[2]),
        pick(&kijkab[3]),
        pick(&kijkab[4]),
    );
    // shape = [np.size(x) for x in kijkab]
    let shape: Vec<usize> = kijkab.iter().map(KIdx::len).collect();
    let (ni, nj, nk, na, nb) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
    let mut out = vec![0_i32; ni * nj * nk * na * nb];

    for (i, kpti) in ki.iter().enumerate() {
        for (j, kptj) in kj.iter().enumerate() {
            for (kx, kptk) in kk.iter().enumerate() {
                for (ax, kpta) in ka.iter().enumerate() {
                    for (bx, kptb) in kb.iter().enumerate() {
                        // kv_ijkab = kk - ka - kb + ki + kj
                        let kv = [
                            kptk[0] - kpta[0] - kptb[0] + kpti[0] + kptj[0],
                            kptk[1] - kpta[1] - kptb[1] + kpti[1] + kptj[1],
                            kptk[2] - kpta[2] - kptb[2] + kpti[2] + kptj[2],
                        ];
                        for (c, kptc) in kpts.iter().enumerate() {
                            let d = [kv[0] - kptc[0], kv[1] - kptc[1], kv[2] - kptc[2]];
                            if is_integral(&frac_component(&a2, &d)) {
                                let idx = ((((i * nj + j) * nk + kx) * na + ax) * nb) + bx;
                                out[idx] = c as i32;
                            }
                        }
                    }
                }
            }
        }
    }

    // new_shape drops every axis whose kijkab entry was a plain integer.
    let new_shape: Vec<usize> = kijkab
        .iter()
        .filter(|x| !x.is_scalar())
        .map(KIdx::len)
        .collect();
    Kconserv3 {
        shape: new_shape,
        data: out,
    }
}

// ---------------------------------------------------------------------------
// Wrap-around unique / conjugation pairing — `kpts_helper.py:144-268`.
// Plan 14-02 Task 2. Phase 13 named these as its own and did not ship them;
// `gdf_builder::gen_uniq_kpts_groups` is the first caller that needs them.
// ---------------------------------------------------------------------------

/// `np.modf(x)[0]` then fold into `[-0.5, 0.5)` — the body shared by
/// [`unique_with_wrap_around`] and `group_by_conj_pairs`.
///
/// `scaled` must already be `.round(5)`-ed by the caller where upstream rounds.
fn fold_to_fbz(scaled: &mut [[f64; 3]]) {
    for k in scaled.iter_mut() {
        for c in k.iter_mut() {
            // np.modf keeps the fractional part WITH the sign of the input.
            *c = c.trunc().mul_add(-1.0, *c);
            if *c >= 0.5 {
                *c -= 1.0;
            } else if *c < -0.5 {
                *c += 1.0;
            }
        }
    }
}

/// `round(x, 5)` — numpy's `.round(5)`, half-to-even like Rust's `round_ties_even`.
fn round5(x: f64) -> f64 {
    (x * 1e5).round_ties_even() / 1e5
}

/// `unique_with_wrap_around(cell, kpts)` — `kpts_helper.py:144-153`.
///
/// Unique k-points **modulo a reciprocal-lattice vector**. `scaled` is the
/// caller's `cell.get_scaled_kpts(kpts)`; keeping the conversion outside means
/// `pyscf-pbc-lib` stays below `pyscf-pbc-gto` in the DAG, exactly as
/// [`get_kconserv`] takes `a` rather than a `Cell`.
///
/// Returns `(index, inverse)` INTO the caller's `kpts`; upstream's `uniq_kpts`
/// is `kpts[index]`, which the caller can form itself.
pub fn unique_with_wrap_around(scaled: &[[f64; 3]]) -> (Vec<usize>, Vec<usize>) {
    let mut s: Vec<[f64; 3]> = scaled
        .iter()
        .map(|k| [round5(k[0]), round5(k[1]), round5(k[2])])
        .collect();
    fold_to_fbz(&mut s);
    let u = unique(&s);
    (u.index, u.inverse)
}

/// One entry of [`group_by_conj_pairs`]: `(k, Some(k_conj))`, with
/// `k == k_conj` for a self-conjugate point, or `(k, None)` when the conjugate
/// is not in the input set.
pub type ConjPair = (usize, Option<usize>);

/// `group_by_conj_pairs(cell, kpts, wrap_around=True, return_kpts_pairs=False)`
/// — `kpts_helper.py:170-218`.
///
/// Three cases, exactly as upstream's docstring lists them:
/// 1. self-conjugate — both indices equal;
/// 2. conjugate present in `kpts` — the pair of indices;
/// 3. conjugate absent — `(index, None)`.
///
/// `scaled` is `cell.get_scaled_kpts(kpts)`. The self-conjugate mask is tested
/// FIRST and its members are emitted first, which fixes the output order.
pub fn group_by_conj_pairs(scaled: &[[f64; 3]], wrap_around: bool) -> Vec<ConjPair> {
    let n = scaled.len();
    let mut s: Vec<[f64; 3]> = scaled.to_vec();
    let mut sc: Vec<[f64; 3]> = scaled.iter().map(|k| [-k[0], -k[1], -k[2]]).collect();
    if wrap_around {
        // Upstream folds with `.round(5) > .5` / `<= -.5`, i.e. the comparison
        // is on the ROUNDED value but the shift lands on the unrounded one.
        for v in [&mut s, &mut sc] {
            for k in v.iter_mut() {
                for c in k.iter_mut() {
                    *c = c.trunc().mul_add(-1.0, *c);
                    let r = round5(*c);
                    if r > 0.5 {
                        *c -= 1.0;
                    } else if r <= -0.5 {
                        *c += 1.0;
                    }
                }
            }
        }
    }
    for v in [&mut s, &mut sc] {
        for k in v.iter_mut() {
            for c in k.iter_mut() {
                *c = round5(*c);
            }
        }
    }

    let self_conj: Vec<bool> = (0..n)
        .map(|k| {
            (0..3)
                .map(|c| (s[k][c] - sc[k][c]).abs())
                .fold(0.0_f64, f64::max)
                < KPT_DIFF_TOL
        })
        .collect();

    let mut pairs: Vec<ConjPair> = (0..n)
        .filter(|k| self_conj[*k])
        .map(|k| (k, Some(k)))
        .collect();
    let mut seen = self_conj;
    for k in 0..n {
        if seen[k] {
            continue;
        }
        seen[k] = true;
        let hits = member(&sc[k], &s);
        match hits.first() {
            None => pairs.push((k, None)),
            Some(&j) => {
                seen[j] = true;
                pairs.push((k, Some(j)));
            }
        }
    }
    pairs
}

/// One yield of [`kk_adapted_iter`].
#[derive(Debug, Clone, PartialEq)]
pub struct KkGroup {
    /// The `ki - kj` difference this group is adapted to, in ABSOLUTE units.
    pub kpt: [f64; 3],
    /// The bra k-point indices, `kpt_ij_idx / nkpts`.
    pub ki_idx: Vec<usize>,
    /// The ket k-point indices, `kpt_ij_idx % nkpts`.
    pub kj_idx: Vec<usize>,
    /// `k == k_conj` — the metric for this group is real.
    pub self_conj: bool,
}

/// `kk_adapted_iter(cell, kpts, kk_idx, time_reversal_symmetry)` —
/// `kpts_helper.py:220-268`.
///
/// Groups the `nkpts²` `(ki, kj)` pairs by their difference `kj - ki`, because
/// the density-fitting metric `j2c` depends only on that difference. Each group
/// is one `get_2c2e` call in `gdf_builder::gen_uniq_kpts_groups`.
///
/// * `scaled_dk` — `cell.get_scaled_kpts(kj - ki)` for every pair, in
///   `ki * nkpts + kj` order (the caller builds it; see `unique_with_wrap_around`).
/// * `dk_abs` — the same differences in absolute units, for the `kpt` field.
/// * `kk_idx` — an optional subset of pair indices. Upstream raises when it is
///   combined with `time_reversal_symmetry`; so does this port.
///
/// # Errors
/// `Err(())` for the `kk_idx` + `time_reversal_symmetry` combination upstream
/// refuses.
#[allow(clippy::result_unit_err)]
pub fn kk_adapted_iter(
    nkpts: usize,
    scaled_dk: &[[f64; 3]],
    dk_abs: &[[f64; 3]],
    kk_idx: Option<&[usize]>,
    time_reversal_symmetry: bool,
) -> Result<Vec<KkGroup>, ()> {
    if kk_idx.is_some() && time_reversal_symmetry {
        return Err(());
    }
    let (uniq_index, uniq_inverse) = unique_with_wrap_around(scaled_dk);
    let uniq_scaled: Vec<[f64; 3]> = uniq_index.iter().map(|&i| scaled_dk[i]).collect();
    let uniq_abs: Vec<[f64; 3]> = uniq_index.iter().map(|&i| dk_abs[i]).collect();
    let groups = group_by_conj_pairs(&uniq_scaled, true);

    let rows = |u: usize| -> Vec<usize> {
        match kk_idx {
            None => (0..uniq_inverse.len())
                .filter(|&p| uniq_inverse[p] == u)
                .collect(),
            Some(sel) => sel
                .iter()
                .enumerate()
                .filter(|(p, _)| uniq_inverse[*p] == u)
                .map(|(_, &v)| v)
                .collect(),
        }
    };

    let mut out = Vec::new();
    for (k, k_conj) in groups {
        let self_conj = k_conj == Some(k);
        let idx = rows(k);
        out.push(KkGroup {
            kpt: uniq_abs[k],
            ki_idx: idx.iter().map(|p| p / nkpts).collect(),
            kj_idx: idx.iter().map(|p| p % nkpts).collect(),
            self_conj,
        });

        if self_conj || k_conj.is_none() || time_reversal_symmetry {
            continue;
        }
        let kc = k_conj.unwrap_or(k);
        let idx = rows(kc);
        out.push(KkGroup {
            kpt: uniq_abs[kc],
            ki_idx: idx.iter().map(|p| p / nkpts).collect(),
            kj_idx: idx.iter().map(|p| p % nkpts).collect(),
            self_conj,
        });
    }
    Ok(out)
}

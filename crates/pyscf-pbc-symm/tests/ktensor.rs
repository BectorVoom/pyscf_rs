//! `pyscf-pbc-symm::ktensor` — 17-06-PLAN.md Tasks 1-4.
//!
//! # Why a round-trip test is NOT enough (the plan's Task-2 warning)
//!
//! `set_4d` decides which of the `nkpts^3` blocks is stored, and
//! `transform_4d` decides which stored block a non-stored key reads from.
//! **They use the same map**, so a round-trip through
//! `set_4d` -> `get_4d` cannot see a wrong one. Every index-map assertion
//! here is therefore against an INDEPENDENTLY built dense tensor
//! ([`set_4d_stores_each_triple_where_an_independent_dense_tensor_says`]),
//! exactly as `15-CONTEXT` insisted for the `ao2mo_7d` index contract.
//!
//! The same applies to the contraction itself:
//! [`transform_4d_matches_an_independent_einsum_for_every_label_and_trans`]
//! compares the four chained `zgemm`s against a direct
//! `sum_{ijab} A[ijab] ri[ik] rj[jl] ra[ac] rb[bd]` written out in this
//! file, not against another call to the same code.
//!
//! # The `trans` flag is enumerated, never inferred
//!
//! Time reversal is antiunitary, so a transformed block is conjugated.
//! `14-VERIFICATION` recorded that defect class twice
//! (`gen_uniq_kpts_groups`'s missing `if self_conj: j2c = j2c.real` and its
//! missing `_conj_j2c` pass — both invisible on the one route that happened
//! to be tested). Every `(label, trans)` combination gets its own
//! comparison: 4 x 4 = 16 for rank 2, 16 x 16 = 256 for rank 4.
//!
//! # Every assert reports the MAXIMUM residual
//!
//! `17-04-MEASUREMENT.md`: a first-violation assert reported 1.58e-11 while
//! the true maximum was 3.99e-10, a 25x difference that changed the
//! diagnosis. [`Worst`] tracks the max and prints it under `--nocapture`.

#![allow(clippy::needless_range_loop)]

use num_complex::Complex64;

use pyscf_pbc_gto::Cell;
use pyscf_pbc_gto::make_kpts_default;
use pyscf_pbc_gto::test_systems::si_precision;
use pyscf_pbc_symm::kpts::{KPoints, KQuartets, MORotationMatrix, make_kpts};
use pyscf_pbc_symm::ktensor::{
    Blocks, Conj, FlatBlocks, Key, KsymmArray, KsymmMeta, OrbSpace, SliceSpec, SubarrayOrder,
    index_to_coords, parse_label, parse_trans, set_2d, set_4d, slice_to_coords, transform_2d,
    transform_4d,
};
use pyscf_pbc_symm::symmetry::build_lattice_symmetry;

const C0: Complex64 = Complex64::new(0.0, 0.0);

/// The oracle-comparison floor. The chained-`zgemm` route and the direct
/// `sum_{ijab}` oracle sum the SAME O(1) terms in different orders, so the
/// residual is pure floating-point reassociation on at most `3^4 = 81`
/// terms — machine epsilon scale.
const EINSUM_TOL: f64 = 1e-13;

// ---------------------------------------------------------------------
// Worst-element tracking (17-04-MEASUREMENT.md's lesson)
// ---------------------------------------------------------------------

struct Worst {
    val: f64,
    at: String,
}

impl Worst {
    fn new() -> Self {
        Worst { val: 0.0, at: String::new() }
    }
    fn see(&mut self, val: f64, at: impl FnOnce() -> String) {
        if val > self.val {
            self.val = val;
            self.at = at();
        }
    }
    fn report(&self, what: &str, tol: f64) {
        println!("  max {what:<56} = {:e}   (tol {tol:e}, at {})", self.val, self.at);
        assert!(
            self.val < tol,
            "max {what} = {:e} exceeds {tol:e} at {}",
            self.val,
            self.at
        );
    }
}

// ---------------------------------------------------------------------
// Deterministic synthetic data
// ---------------------------------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let x = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        2.0 * x - 1.0
    }
    fn next_c(&mut self) -> Complex64 {
        Complex64::new(self.next_f64(), self.next_f64())
    }
}

/// A deterministic complex `n x n` UNITARY, by modified Gram-Schmidt on the
/// rows of a pseudo-random matrix. Unitarity is what makes the Hermiticity
/// invariant meaningful; the einsum oracle does not need it.
fn unitary(n: usize, rng: &mut Rng) -> Vec<Complex64> {
    let mut m: Vec<Complex64> = (0..n * n).map(|_| rng.next_c()).collect();
    for i in 0..n {
        for j in 0..i {
            // <row_j, row_i>
            let mut ip = C0;
            for p in 0..n {
                ip += m[j * n + p].conj() * m[i * n + p];
            }
            for p in 0..n {
                let v = m[j * n + p];
                m[i * n + p] -= ip * v;
            }
        }
        let mut nrm = 0.0;
        for p in 0..n {
            nrm += m[i * n + p].norm_sqr();
        }
        let nrm = nrm.sqrt();
        for p in 0..n {
            m[i * n + p] /= nrm;
        }
    }
    m
}

fn identity(n: usize) -> Vec<Complex64> {
    let mut m = vec![C0; n * n];
    for i in 0..n {
        m[i * n + i] = Complex64::new(1.0, 0.0);
    }
    m
}

/// A synthetic [`MORotationMatrix`] with `nkpts x nops` unitaries per block.
/// `MORotationMatrix`'s fields are `pub`, so no SCF is needed to exercise
/// the contraction algebra.
fn synthetic_rmat(nkpts: usize, nops: usize, nocc: usize, nvir: usize, seed: u64) -> MORotationMatrix {
    let mut rng = Rng::new(seed);
    let mut r = MORotationMatrix::new(nocc, nocc + nvir);
    r.oo = Some(
        (0..nkpts).map(|_| (0..nops).map(|_| unitary(nocc, &mut rng)).collect()).collect(),
    );
    r.vv = Some(
        (0..nkpts).map(|_| (0..nops).map(|_| unitary(nvir, &mut rng)).collect()).collect(),
    );
    r
}

/// The same shape, but every rotation is the IDENTITY — the oracle-free
/// invariant `transform_*(block, identity) == block`.
fn identity_rmat(nkpts: usize, nops: usize, nocc: usize, nvir: usize) -> MORotationMatrix {
    let mut r = MORotationMatrix::new(nocc, nocc + nvir);
    r.oo = Some((0..nkpts).map(|_| (0..nops).map(|_| identity(nocc)).collect()).collect());
    r.vv = Some((0..nkpts).map(|_| (0..nops).map(|_| identity(nvir)).collect()).collect());
    r
}

// ---------------------------------------------------------------------
// Independent oracles — written out here, never a second call into the port
// ---------------------------------------------------------------------

fn ref_conj(m: &[Complex64], t: Conj) -> Vec<Complex64> {
    match t {
        Conj::N => m.to_vec(),
        Conj::C => m.iter().map(|z| z.conj()).collect(),
    }
}

/// `out[k,l] = sum_{i,j} a[i,j] ri[i,k] rj[j,l]` — upstream's
/// `reduce(np.dot, (rot_i.T, arr, rot_j))` (`ktensor.py:286`), written as a
/// direct summation.
fn ref_2d(a: &[Complex64], di: usize, dj: usize, ri: &[Complex64], rj: &[Complex64]) -> Vec<Complex64> {
    let mut out = vec![C0; di * dj];
    for k in 0..di {
        for l in 0..dj {
            let mut s = C0;
            for i in 0..di {
                for j in 0..dj {
                    s += a[i * dj + j] * ri[i * di + k] * rj[j * dj + l];
                }
            }
            out[k * dj + l] = s;
        }
    }
    out
}

/// `out[k,l,c,d] = sum_{i,j,a,b} arr[i,j,a,b] ri[i,k] rj[j,l] ra[a,c] rb[b,d]`
/// — upstream's commented-out `einsum('ijab,ik,jl,ac,bd->klcd', ...)`
/// (`ktensor.py:319-320`), written as a direct summation.
#[allow(clippy::too_many_arguments)]
fn ref_4d(
    arr: &[Complex64],
    d: [usize; 4],
    ri: &[Complex64],
    rj: &[Complex64],
    ra: &[Complex64],
    rb: &[Complex64],
) -> Vec<Complex64> {
    let [di, dj, da, db] = d;
    let mut out = vec![C0; di * dj * da * db];
    for k in 0..di {
        for l in 0..dj {
            for c in 0..da {
                for dd in 0..db {
                    let mut s = C0;
                    for i in 0..di {
                        for j in 0..dj {
                            for a in 0..da {
                                for b in 0..db {
                                    s += arr[((i * dj + j) * da + a) * db + b]
                                        * ri[i * di + k]
                                        * rj[j * dj + l]
                                        * ra[a * da + c]
                                        * rb[b * db + dd];
                                }
                            }
                        }
                    }
                    out[((k * dj + l) * da + c) * db + dd] = s;
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------
// The symmetry fixture — NO SCF. `si` at [2,2,2], full space group.
// ---------------------------------------------------------------------

struct Sym {
    /// Kept so the fixture owns the `Cell` its `KQuartets` was built from —
    /// nothing reads it afterwards (`KQuartets` borrows a `&Cell` per call,
    /// 17-CONTEXT §3.9), but dropping it would make the fixture's provenance
    /// invisible.
    #[allow(dead_code)]
    cell: Cell,
    kpts: KPoints,
    kqrts: KQuartets,
}

fn sym() -> &'static Sym {
    static S: std::sync::OnceLock<Sym> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        let mut cell = si_precision(1e-8);
        cell.space_group_symmetry = true;
        cell.symmorphic = false;
        let check_mesh_symmetry = !cell._mesh_from_build;
        build_lattice_symmetry(&mut cell, check_mesh_symmetry).expect("build_lattice_symmetry");
        let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
        let kpts = make_kpts(&cell, &kpts_abs, true, true).expect("make_kpts");
        assert!(!kpts.time_reversal, "si has inversion: time reversal must be OFF");
        let kqrts = KQuartets::build(&kpts, &cell).expect("KQuartets::build");
        Sym { cell, kpts, kqrts }
    })
}

fn n_ops(kpts: &KPoints) -> usize {
    kpts.k2opk.first().map_or(0, |r| r.len())
}

/// A FULL-BZ k-point that is NOT its own IBZ representative — the only kind
/// for which `transform_2d` takes the contraction branch.
fn a_non_representative_k(kpts: &KPoints) -> usize {
    (0..kpts.nkpts())
        .find(|&k| kpts.ibz2bz[kpts.bz2ibz[k]] != k)
        .expect("si [2,2,2] must have a reducible k-point")
}

/// A FULL-BZ triple that is NOT its own `kqrts` representative.
fn a_non_representative_triple(kpts: &KPoints, kqrts: &KQuartets) -> [usize; 3] {
    let n = kpts.nkpts();
    for i in 0..n {
        for j in 0..n {
            for a in 0..n {
                let kk = kpts.ktuple_to_index(&[i, j, a]);
                let q = kqrts.kqrts_ibz[kqrts.bz2ibz[kk]];
                if [q[0], q[1], q[2]] != [i, j, a] {
                    return [i, j, a];
                }
            }
        }
    }
    panic!("si [2,2,2] must have a reducible k-triple");
}

// =====================================================================
// Task 2 — the index algebra, EXHAUSTIVELY (ktensor.py:339-381)
// =====================================================================

/// A reference `np.arange(start, stop, step)` written independently of
/// [`slice_to_coords`], plus upstream's own `None`/negative folding
/// (`ktensor.py:370-380`).
fn ref_slice(spec: SliceSpec, n: i64) -> Vec<i64> {
    let start = match spec.start {
        None => 0,
        Some(s) if s < 0 => s + n,
        Some(s) => s,
    };
    let stop = match spec.stop {
        None => n,
        Some(s) if s < 0 => s + n,
        Some(s) => s,
    };
    let step = spec.step.unwrap_or(1);
    let mut out = Vec::new();
    if step > 0 {
        let mut v = start;
        while v < stop {
            out.push(v);
            v += step;
        }
    } else {
        let mut v = start;
        while v > stop {
            out.push(v);
            v += step;
        }
    }
    out
}

#[test]
fn slice_to_coords_is_exhaustively_numpy_arange() {
    // EXHAUSTIVE over the whole cross product, not a sample: every
    // start/stop in [-2n, 2n] plus None, every step in [-n, n] \ {0} plus
    // None, for n = 1..6.
    let mut cases = 0usize;
    for n in 1i64..=6 {
        let bounds: Vec<Option<i64>> =
            std::iter::once(None).chain((-2 * n..=2 * n).map(Some)).collect();
        let steps: Vec<Option<i64>> =
            std::iter::once(None).chain((-n..=n).filter(|s| *s != 0).map(Some)).collect();
        for &start in &bounds {
            for &stop in &bounds {
                for &step in &steps {
                    let spec = SliceSpec { start, stop, step };
                    let got = slice_to_coords(spec, n as usize).expect("slice_to_coords");
                    assert_eq!(got, ref_slice(spec, n), "n = {n}, spec = {spec:?}");
                    cases += 1;
                }
            }
        }
    }
    println!("  slice_to_coords: {cases} exhaustive cases");
    assert!(cases > 5000, "the sweep must be exhaustive, not a sample");

    // step == 0 is NumPy's ZeroDivisionError.
    assert!(slice_to_coords(SliceSpec { start: None, stop: None, step: Some(0) }, 4).is_err());
}

/// A reference `lib.cartesian_prod` — LAST axis varying fastest.
fn ref_cartesian(idxs: &[Vec<i64>]) -> Vec<Vec<i64>> {
    let mut out = vec![vec![]];
    for axis in idxs {
        let mut next = Vec::with_capacity(out.len() * axis.len());
        for row in &out {
            for &v in axis {
                let mut r = row.clone();
                r.push(v);
                next.push(r);
            }
        }
        out = next;
    }
    out
}

#[test]
fn index_to_coords_is_exhaustive_over_a_full_generator_set_at_small_nkpts() {
    let n = 3usize;
    let shape = [n, n, n];
    // The generator set at each axis: every integer, several slices, an
    // explicit array. 3 + 4 + 1 = 8 choices per axis.
    let gens: Vec<Key> = (0..n as i64)
        .map(Key::Index)
        .chain([
            Key::Slice(SliceSpec::full()),
            Key::Slice(SliceSpec { start: Some(1), stop: None, step: None }),
            Key::Slice(SliceSpec { start: None, stop: Some(-1), step: None }),
            Key::Slice(SliceSpec { start: None, stop: None, step: Some(2) }),
            Key::Array(vec![2, 0]),
        ])
        .collect();

    let mut cases = 0usize;
    // key lengths 0, 1, 2 and 3 — the WHOLE cross product of the generator
    // set at each position, not a sample.
    for len in 0..=3usize {
        let total = gens.len().pow(len as u32);
        for enc in 0..total {
            let mut e = enc;
            let mut key: Vec<Key> = Vec::with_capacity(len);
            for _ in 0..len {
                key.push(gens[e % gens.len()].clone());
                e /= gens.len();
            }
            // Independent expectation.
            let mut idxs: Vec<Vec<i64>> = Vec::new();
            for (d, k) in key.iter().enumerate() {
                idxs.push(match k {
                    Key::Index(v) => vec![*v],
                    Key::Slice(sp) => ref_slice(*sp, shape[d] as i64),
                    Key::Array(v) => v.clone(),
                });
            }
            for d in len..3 {
                idxs.push((0..shape[d] as i64).collect());
            }
            let expect = ref_cartesian(&idxs);
            let all_int = key.iter().all(|k| matches!(k, Key::Index(_)));

            let got = index_to_coords(&key, &shape).expect("index_to_coords");
            let rows: Vec<Vec<i64>> = got.rows().iter().map(|r| r.to_vec()).collect();
            if all_int && len == 3 {
                // `:365-366` — the ONLY case that collapses to `ndim == 1`.
                assert!(
                    matches!(got, pyscf_pbc_symm::ktensor::Coords::Single(_)),
                    "full-rank all-integer key must collapse: {key:?}"
                );
                assert_eq!(rows, vec![expect[0].clone()]);
            } else {
                assert!(
                    matches!(got, pyscf_pbc_symm::ktensor::Coords::Many(_)),
                    "key {key:?} (len {len}) must NOT collapse"
                );
                assert_eq!(rows, expect, "key = {key:?}");
            }
            cases += 1;
        }
    }
    println!("  index_to_coords: {cases} exhaustive cases over an 8-element generator set");
    assert!(cases > 500, "the sweep must be exhaustive, not a sample");

    // A key longer than the shape is upstream's bare `raise RuntimeError`.
    assert!(index_to_coords(&[Key::Index(0), Key::Index(0), Key::Index(0), Key::Index(0)], &shape).is_err());
}

// =====================================================================
// Task 2 — set_2d / set_4d against an INDEPENDENT dense tensor
// =====================================================================

/// A distinctive deterministic block value: the triple is recoverable from
/// any single element, so a block stored at the wrong slot is visible by
/// inspection, not only by a residual.
fn dense_block_4d(ki: usize, kj: usize, ka: usize, block_len: usize) -> Vec<Complex64> {
    (0..block_len)
        .map(|p| {
            Complex64::new(
                (ki * 1_000_000 + kj * 10_000 + ka * 100 + p) as f64,
                -((ki * 7 + kj * 5 + ka * 3 + p) as f64),
            )
        })
        .collect()
}

fn dense_block_2d(ki: usize, block_len: usize) -> Vec<Complex64> {
    (0..block_len)
        .map(|p| Complex64::new((ki * 100 + p) as f64, -((ki * 3 + p) as f64)))
        .collect()
}

#[test]
fn set_4d_stores_each_triple_where_an_independent_dense_tensor_says() {
    let s = sym();
    let (kpts, kqrts) = (&s.kpts, &s.kqrts);
    let n = kpts.nkpts();
    let block_len = 4usize;
    let nb = kqrts.kqrts_ibz.len();

    // Write EVERY full-BZ triple. `set_4d` must keep exactly the
    // representatives and discard the rest.
    let mut triples: Vec<[usize; 3]> = Vec::with_capacity(n * n * n);
    let mut blocks: Vec<Vec<Complex64>> = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for a in 0..n {
                triples.push([i, j, a]);
                blocks.push(dense_block_4d(i, j, a, block_len));
            }
        }
    }
    let vals: Vec<&[Complex64]> = blocks.iter().map(|b| b.as_slice()).collect();

    let mut data = vec![C0; nb * block_len];
    {
        let mut sink = FlatBlocks::new(&mut data, block_len).expect("FlatBlocks");
        set_4d(&mut sink, &vals, kpts, kqrts, &triples).expect("set_4d");
    }

    // The INDEPENDENT expectation: slot `m` must hold the dense block of
    // `kqrts.kqrts_ibz[m][:3]`, derived straight from `kqrts_ibz` without
    // touching `ktuple_to_index` or `bz2ibz`.
    let mut expect = vec![C0; nb * block_len];
    for (m, q) in kqrts.kqrts_ibz.iter().enumerate() {
        let b = dense_block_4d(q[0], q[1], q[2], block_len);
        expect[m * block_len..(m + 1) * block_len].copy_from_slice(&b);
    }

    assert_eq!(nb, kqrts.kqrts_ibz.len());
    println!("  set_4d: nkpts = {n}, nkpts^3 = {}, stored blocks = {nb}", n * n * n);
    for m in 0..nb {
        for p in 0..block_len {
            assert_eq!(
                data[m * block_len + p].to_bits_pair(),
                expect[m * block_len + p].to_bits_pair(),
                "slot {m} element {p}: stored {} but the dense tensor says {} \
                 (kqrts_ibz[{m}] = {:?})",
                data[m * block_len + p],
                expect[m * block_len + p],
                kqrts.kqrts_ibz[m]
            );
        }
    }
}

#[test]
fn set_2d_stores_each_bz_key_at_its_ibz_slot_and_discards_the_rest() {
    let s = sym();
    let kpts = &s.kpts;
    let n = kpts.nkpts();
    let block_len = 3usize;
    let nb = kpts.nkpts_ibz();

    let keys: Vec<usize> = (0..n).collect();
    let blocks: Vec<Vec<Complex64>> = keys.iter().map(|&k| dense_block_2d(k, block_len)).collect();
    let vals: Vec<&[Complex64]> = blocks.iter().map(|b| b.as_slice()).collect();

    let mut data = vec![C0; nb * block_len];
    {
        let mut sink = FlatBlocks::new(&mut data, block_len).expect("FlatBlocks");
        set_2d(&mut sink, &vals, kpts, &keys).expect("set_2d");
    }

    // Independent expectation from `ibz2bz` alone.
    let mut expect = vec![C0; nb * block_len];
    for (m, &k) in kpts.ibz2bz.iter().enumerate() {
        expect[m * block_len..(m + 1) * block_len].copy_from_slice(&dense_block_2d(k, block_len));
    }
    println!("  set_2d: nkpts = {n}, nkpts_ibz = {nb}, ibz2bz = {:?}", kpts.ibz2bz);
    assert_eq!(data, expect);
}

/// Bit-comparison helper — a `Complex64` pair of `to_bits()`.
trait BitsPair {
    fn to_bits_pair(&self) -> (u64, u64);
}
impl BitsPair for Complex64 {
    fn to_bits_pair(&self) -> (u64, u64) {
        (self.re.to_bits(), self.im.to_bits())
    }
}

// =====================================================================
// Task 3 — transform_2d / transform_4d, EVERY (label, trans)
// =====================================================================

const SPACES: [OrbSpace; 2] = [OrbSpace::Occ, OrbSpace::Vir];
const CONJS: [Conj; 2] = [Conj::N, Conj::C];

const NOCC: usize = 2;
const NVIR: usize = 3;

fn dim_of(sp: OrbSpace) -> usize {
    match sp {
        OrbSpace::Occ => NOCC,
        OrbSpace::Vir => NVIR,
    }
}

fn rot_ref<'r>(rmat: &'r MORotationMatrix, sp: OrbSpace, k: usize, iop: usize) -> &'r [Complex64] {
    match sp {
        OrbSpace::Occ => &rmat.oo.as_ref().expect("oo")[k][iop],
        OrbSpace::Vir => &rmat.vv.as_ref().expect("vv")[k][iop],
    }
}

#[test]
fn transform_2d_matches_an_independent_einsum_for_every_label_and_trans() {
    let s = sym();
    let kpts = &s.kpts;
    let nops = n_ops(kpts);
    let rmat = synthetic_rmat(kpts.nkpts(), nops, NOCC, NVIR, 170_602);
    let ki = a_non_representative_k(kpts);
    let ki_ibz = kpts.bz2ibz[ki];
    let ki_ibz_bz = kpts.ibz2bz[ki_ibz];
    let iop = kpts.stars_ops_bz[ki];
    println!("  transform_2d: ki = {ki}, rep = {ki_ibz_bz} (slot {ki_ibz}), iop = {iop}");

    let mut rng = Rng::new(0xC0FFEE);
    let mut worst = Worst::new();
    let mut combos = 0usize;
    for &pi in &SPACES {
        for &pj in &SPACES {
            let (di, dj) = (dim_of(pi), dim_of(pj));
            // One stored block per (label) — every stored slot filled so a
            // wrong `ki_ibz` reads different numbers.
            let flat: Vec<Complex64> =
                (0..kpts.nkpts_ibz() * di * dj).map(|_| rng.next_c()).collect();
            let blocks = Blocks::new(&flat, di * dj).expect("Blocks");
            let stored = blocks.block(ki_ibz).expect("block");

            for &ti in &CONJS {
                for &tj in &CONJS {
                    let got = transform_2d(
                        &blocks,
                        kpts,
                        ki,
                        &rmat,
                        &[pi, pj],
                        &[ti, tj],
                        [di, dj],
                    )
                    .expect("transform_2d");
                    let ri = ref_conj(rot_ref(&rmat, pi, ki_ibz_bz, iop), ti);
                    let rj = ref_conj(rot_ref(&rmat, pj, ki_ibz_bz, iop), tj);
                    let want = ref_2d(stored, di, dj, &ri, &rj);
                    assert_eq!(got.len(), want.len());
                    for p in 0..got.len() {
                        worst.see((got[p] - want[p]).norm(), || {
                            format!("label {pi:?}{pj:?} trans {ti:?}{tj:?} elem {p}")
                        });
                    }
                    combos += 1;
                }
            }
        }
    }
    println!("  transform_2d: {combos} (label, trans) combinations");
    assert_eq!(combos, 16, "every (label, trans) combination must be tested");
    worst.report("|transform_2d - independent einsum|", EINSUM_TOL);
}

#[test]
fn transform_4d_matches_an_independent_einsum_for_every_label_and_trans() {
    let s = sym();
    let (kpts, kqrts) = (&s.kpts, &s.kqrts);
    let nops = n_ops(kpts);
    let rmat = synthetic_rmat(kpts.nkpts(), nops, NOCC, NVIR, 170_604);
    let klc = a_non_representative_triple(kpts, kqrts);
    let kk_bz = kpts.ktuple_to_index(&klc);
    let kk_ibz = kqrts.bz2ibz[kk_bz];
    let q = kqrts.kqrts_ibz[kk_ibz];
    let iop = kqrts.stars_ops_bz[kk_bz];
    println!("  transform_4d: klc = {klc:?}, rep = {q:?} (slot {kk_ibz}), iop = {iop}");

    let mut rng = Rng::new(0xBEEF_1706);
    let mut worst = Worst::new();
    let mut combos = 0usize;
    for &pi in &SPACES {
        for &pj in &SPACES {
            for &pa in &SPACES {
                for &pb in &SPACES {
                    let d = [dim_of(pi), dim_of(pj), dim_of(pa), dim_of(pb)];
                    let bl: usize = d.iter().product();
                    let flat: Vec<Complex64> =
                        (0..kqrts.kqrts_ibz.len() * bl).map(|_| rng.next_c()).collect();
                    let blocks = Blocks::new(&flat, bl).expect("Blocks");
                    let stored = blocks.block(kk_ibz).expect("block");

                    for &ti in &CONJS {
                        for &tj in &CONJS {
                            for &ta in &CONJS {
                                for &tb in &CONJS {
                                    let got = transform_4d(
                                        &blocks,
                                        kpts,
                                        kqrts,
                                        klc,
                                        &rmat,
                                        &[pi, pj, pa, pb],
                                        &[ti, tj, ta, tb],
                                        d,
                                    )
                                    .expect("transform_4d");
                                    let ri = ref_conj(rot_ref(&rmat, pi, q[0], iop), ti);
                                    let rj = ref_conj(rot_ref(&rmat, pj, q[1], iop), tj);
                                    let ra = ref_conj(rot_ref(&rmat, pa, q[2], iop), ta);
                                    let rb = ref_conj(rot_ref(&rmat, pb, q[3], iop), tb);
                                    let want = ref_4d(stored, d, &ri, &rj, &ra, &rb);
                                    assert_eq!(got.len(), want.len());
                                    for p in 0..got.len() {
                                        worst.see((got[p] - want[p]).norm(), || {
                                            format!(
                                                "label {pi:?}{pj:?}{pa:?}{pb:?} \
                                                 trans {ti:?}{tj:?}{ta:?}{tb:?} elem {p}"
                                            )
                                        });
                                    }
                                    combos += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    println!("  transform_4d: {combos} (label, trans) combinations");
    assert_eq!(combos, 256, "every (label, trans) combination must be tested");
    worst.report("|transform_4d - independent einsum|", EINSUM_TOL);
}

#[test]
fn transform_with_an_identity_rotation_returns_the_block_bit_exactly() {
    let s = sym();
    let (kpts, kqrts) = (&s.kpts, &s.kqrts);
    let nops = n_ops(kpts);
    let rmat = identity_rmat(kpts.nkpts(), nops, NOCC, NVIR);
    let mut rng = Rng::new(0x1DE0);

    // --- rank 2, at a k-point that takes the CONTRACTION branch ---------
    let ki = a_non_representative_k(kpts);
    let ki_ibz = kpts.bz2ibz[ki];
    let (di, dj) = (NOCC, NVIR);
    let flat: Vec<Complex64> = (0..kpts.nkpts_ibz() * di * dj).map(|_| rng.next_c()).collect();
    let blocks = Blocks::new(&flat, di * dj).expect("Blocks");
    let stored = blocks.block(ki_ibz).expect("block").to_vec();
    for &ti in &CONJS {
        for &tj in &CONJS {
            // conj(I) == I, so every `trans` must give the block back.
            let got = transform_2d(
                &blocks,
                kpts,
                ki,
                &rmat,
                &[OrbSpace::Occ, OrbSpace::Vir],
                &[ti, tj],
                [di, dj],
            )
            .expect("transform_2d");
            for p in 0..got.len() {
                assert_eq!(
                    got[p].to_bits_pair(),
                    stored[p].to_bits_pair(),
                    "transform_2d(identity) must be BIT-exact: trans {ti:?}{tj:?} elem {p}"
                );
            }
        }
    }

    // --- rank 4, at a triple that takes the CONTRACTION branch ----------
    let klc = a_non_representative_triple(kpts, kqrts);
    let kk_ibz = kqrts.bz2ibz[kpts.ktuple_to_index(&klc)];
    let d = [NOCC, NOCC, NVIR, NVIR];
    let bl: usize = d.iter().product();
    let flat4: Vec<Complex64> = (0..kqrts.kqrts_ibz.len() * bl).map(|_| rng.next_c()).collect();
    let blocks4 = Blocks::new(&flat4, bl).expect("Blocks");
    let stored4 = blocks4.block(kk_ibz).expect("block").to_vec();
    let label = [OrbSpace::Occ, OrbSpace::Occ, OrbSpace::Vir, OrbSpace::Vir];
    for &ti in &CONJS {
        for &tj in &CONJS {
            for &ta in &CONJS {
                for &tb in &CONJS {
                    let got = transform_4d(
                        &blocks4,
                        kpts,
                        kqrts,
                        klc,
                        &rmat,
                        &label,
                        &[ti, tj, ta, tb],
                        d,
                    )
                    .expect("transform_4d");
                    for p in 0..got.len() {
                        assert_eq!(
                            got[p].to_bits_pair(),
                            stored4[p].to_bits_pair(),
                            "transform_4d(identity) must be BIT-exact: \
                             trans {ti:?}{tj:?}{ta:?}{tb:?} elem {p}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn hermiticity_survives_exactly_the_mixed_trans_combinations() {
    // For a square label (pi == pj) and a UNITARY rotation R:
    //   trans 'cn' -> R^H A R  -> Hermitian when A is
    //   trans 'nc' -> R^T A R* -> Hermitian when A is
    //   trans 'nn' -> R^T A R  -> NOT Hermitian in general
    //   trans 'cc' -> R^H A R* -> NOT Hermitian in general
    // Both halves are asserted: the preserving pair to a tolerance, the
    // breaking pair as LARGE — so nobody can "fix" this test by asserting
    // Hermiticity everywhere.
    let s = sym();
    let kpts = &s.kpts;
    let nops = n_ops(kpts);
    let rmat = synthetic_rmat(kpts.nkpts(), nops, NOCC, NVIR, 170_603);
    let ki = a_non_representative_k(kpts);

    let mut rng = Rng::new(0x4E3_1706);
    for &p in &SPACES {
        let n = dim_of(p);
        // A Hermitian stored block in every slot.
        let mut flat = vec![C0; kpts.nkpts_ibz() * n * n];
        for b in 0..kpts.nkpts_ibz() {
            let m: Vec<Complex64> = (0..n * n).map(|_| rng.next_c()).collect();
            for i in 0..n {
                for j in 0..n {
                    let v = (m[i * n + j] + m[j * n + i].conj()) * 0.5;
                    flat[b * n * n + i * n + j] = v;
                    flat[b * n * n + j * n + i] = v.conj();
                }
            }
        }
        let blocks = Blocks::new(&flat, n * n).expect("Blocks");

        let mut preserved = Worst::new();
        let mut broken_min = f64::INFINITY;
        for &ti in &CONJS {
            for &tj in &CONJS {
                let got =
                    transform_2d(&blocks, kpts, ki, &rmat, &[p, p], &[ti, tj], [n, n])
                        .expect("transform_2d");
                let mut dev: f64 = 0.0;
                for i in 0..n {
                    for j in 0..n {
                        dev = dev.max((got[i * n + j] - got[j * n + i].conj()).norm());
                    }
                }
                let mixed = ti != tj;
                println!(
                    "  hermiticity: label {p:?}{p:?} trans {ti:?}{tj:?} -> |A - A^H| = {dev:e} \
                     ({})",
                    if mixed { "must be ~0" } else { "must be large" }
                );
                if mixed {
                    preserved.see(dev, || format!("trans {ti:?}{tj:?}"));
                } else {
                    broken_min = broken_min.min(dev);
                }
            }
        }
        preserved.report("|A - A^H| for the mixed trans combinations", 1e-14);
        assert!(
            broken_min > 1e-3,
            "trans 'nn'/'cc' are NOT Hermiticity-preserving; the smallest \
             measured deviation was {broken_min:e}. A small value here means \
             the fixture stopped exercising a genuinely complex rotation, not \
             that asserting Hermiticity everywhere became valid."
        );
    }
}

// =====================================================================
// Task 1 — the container: round-trips, incore/outcore, refusals
// =====================================================================

fn meta_2d<'a>(
    kpts: &'a KPoints,
    rmat: &'a MORotationMatrix,
    label: &'a [OrbSpace],
    trans: &'a [Conj],
    incore: bool,
) -> KsymmMeta<'a> {
    KsymmMeta {
        kpts,
        kqrts: None,
        rmat: Some(rmat),
        label: Some(label),
        trans: Some(trans),
        incore,
    }
}

#[allow(clippy::too_many_arguments)]
fn meta_4d<'a>(
    kpts: &'a KPoints,
    kqrts: &'a KQuartets,
    rmat: &'a MORotationMatrix,
    label: &'a [OrbSpace],
    trans: &'a [Conj],
    incore: bool,
) -> KsymmMeta<'a> {
    KsymmMeta {
        kpts,
        kqrts: Some(kqrts),
        rmat: Some(rmat),
        label: Some(label),
        trans: Some(trans),
        incore,
    }
}

#[test]
fn shape_ndim_and_order_accessors_match_upstream() {
    let s = sym();
    let kpts = &s.kpts;
    let kqrts = &s.kqrts;
    let rmat = synthetic_rmat(kpts.nkpts(), n_ops(kpts), NOCC, NVIR, 1);
    let label2 = parse_label("ov", 2).expect("label");
    let trans2 = parse_trans("nc", 2).expect("trans");
    let a = KsymmArray::empty(
        &[NOCC, NVIR],
        SubarrayOrder::C,
        meta_2d(kpts, &rmat, &label2, &trans2, true),
    )
    .expect("empty 2d");
    assert_eq!(a.shape(), vec![kpts.nkpts(), NOCC, NVIR]);
    assert_eq!(a.ndim(), 3);
    assert_eq!(a.subarray_ndim(), 2);
    assert_eq!(a.subarray_shape(), &[NOCC, NVIR]);
    assert_eq!(a.subarray_order(), SubarrayOrder::C);
    assert_eq!(a.n_blocks(), kpts.nkpts_ibz());

    let label4 = parse_label("oovv", 4).expect("label");
    let trans4 = parse_trans("nncc", 4).expect("trans");
    let b = KsymmArray::empty(
        &[NOCC, NOCC, NVIR, NVIR],
        SubarrayOrder::F,
        meta_4d(kpts, kqrts, &rmat, &label4, &trans4, true),
    )
    .expect("empty 4d");
    assert_eq!(b.shape(), vec![kpts.nkpts(), kpts.nkpts(), kpts.nkpts(), NOCC, NOCC, NVIR, NVIR]);
    assert_eq!(b.ndim(), 7);
    assert_eq!(b.subarray_order(), SubarrayOrder::F);
    assert_eq!(b.n_blocks(), kqrts.kqrts_ibz.len());

    // `empty_like` copies shape, order and metadata (`ktensor.py:32-39`).
    let c = KsymmArray::empty_like(&b).expect("empty_like");
    assert_eq!(c.shape(), b.shape());
    assert_eq!(c.subarray_order(), b.subarray_order());
    assert_eq!(c.n_blocks(), b.n_blocks());

    // A rank other than 2 or 4 is upstream's `NotImplementedError`.
    assert!(
        KsymmArray::empty(&[2, 2, 2], SubarrayOrder::C, meta_2d(kpts, &rmat, &label2, &trans2, true))
            .is_err()
    );
    // Bad label / trans strings.
    assert!(parse_label("ox", 2).is_err());
    assert!(parse_label("oov", 2).is_err());
    assert!(parse_trans("nx", 2).is_err());
}

fn fill_dense_2d(kpts: &KPoints, di: usize, dj: usize) -> Vec<Complex64> {
    let mut rng = Rng::new(0xD2);
    (0..kpts.nkpts() * di * dj).map(|_| rng.next_c()).collect()
}

fn fill_dense_4d(kpts: &KPoints, bl: usize) -> Vec<Complex64> {
    let n = kpts.nkpts();
    let mut rng = Rng::new(0xD4);
    (0..n * n * n * bl).map(|_| rng.next_c()).collect()
}

#[test]
fn from_dense_to_dense_round_trips_bit_exactly() {
    let s = sym();
    let (kpts, kqrts) = (&s.kpts, &s.kqrts);
    let rmat = synthetic_rmat(kpts.nkpts(), n_ops(kpts), NOCC, NVIR, 5);

    // --- rank 2 ---------------------------------------------------------
    let label2 = parse_label("oo", 2).expect("l");
    let trans2 = parse_trans("cn", 2).expect("t");
    let dense = fill_dense_2d(kpts, NOCC, NOCC);
    let a = KsymmArray::from_dense(
        &dense,
        &[NOCC, NOCC],
        SubarrayOrder::C,
        meta_2d(kpts, &rmat, &label2, &trans2, true),
    )
    .expect("from_dense 2d");
    // The IBZ blocks must be the dense ones, untouched.
    for (m, &k) in kpts.ibz2bz.iter().enumerate() {
        let got = a.stored_block(m).expect("stored");
        let want = &dense[k * NOCC * NOCC..(k + 1) * NOCC * NOCC];
        assert_eq!(got, want, "IBZ slot {m} (BZ {k})");
    }
    let d2 = a.to_dense().expect("to_dense");
    assert_eq!(d2.len(), kpts.nkpts() * NOCC * NOCC);
    let a2 = KsymmArray::from_dense(
        &d2,
        &[NOCC, NOCC],
        SubarrayOrder::C,
        meta_2d(kpts, &rmat, &label2, &trans2, true),
    )
    .expect("from_dense round 2");
    for m in 0..a.n_blocks() {
        let x = a.stored_block(m).expect("x");
        let y = a2.stored_block(m).expect("y");
        for p in 0..x.len() {
            assert_eq!(x[p].to_bits_pair(), y[p].to_bits_pair(), "2d slot {m} elem {p}");
        }
    }

    // --- rank 4 ---------------------------------------------------------
    let label4 = parse_label("oovv", 4).expect("l");
    let trans4 = parse_trans("nncc", 4).expect("t");
    let d = [NOCC, NOCC, NVIR, NVIR];
    let bl: usize = d.iter().product();
    let dense4 = fill_dense_4d(kpts, bl);
    let b = KsymmArray::from_dense(
        &dense4,
        &d,
        SubarrayOrder::C,
        meta_4d(kpts, kqrts, &rmat, &label4, &trans4, true),
    )
    .expect("from_dense 4d");
    let n = kpts.nkpts();
    for (m, q) in kqrts.kqrts_ibz.iter().enumerate() {
        let off = ((q[0] * n + q[1]) * n + q[2]) * bl;
        assert_eq!(b.stored_block(m).expect("stored"), &dense4[off..off + bl], "4d slot {m}");
    }
    let d4 = b.to_dense().expect("to_dense");
    assert_eq!(d4.len(), n * n * n * bl);
    let b2 = KsymmArray::from_dense(
        &d4,
        &d,
        SubarrayOrder::C,
        meta_4d(kpts, kqrts, &rmat, &label4, &trans4, true),
    )
    .expect("from_dense round 2");
    for m in 0..b.n_blocks() {
        let x = b.stored_block(m).expect("x");
        let y = b2.stored_block(m).expect("y");
        for p in 0..x.len() {
            assert_eq!(x[p].to_bits_pair(), y[p].to_bits_pair(), "4d slot {m} elem {p}");
        }
    }
}

#[test]
fn from_raw_to_raw_round_trips_for_both_subarray_orders() {
    let s = sym();
    let kpts = &s.kpts;
    let rmat = synthetic_rmat(kpts.nkpts(), n_ops(kpts), NOCC, NVIR, 6);
    let label = parse_label("ov", 2).expect("l");
    let trans = parse_trans("nc", 2).expect("t");
    let bl = NOCC * NVIR;
    let mut rng = Rng::new(0x9AB_1706);
    let raw: Vec<Complex64> = (0..kpts.nkpts_ibz() * bl).map(|_| rng.next_c()).collect();

    for order in [SubarrayOrder::C, SubarrayOrder::F] {
        let a = KsymmArray::from_raw(
            &raw,
            &[NOCC, NVIR],
            order,
            meta_2d(kpts, &rmat, &label, &trans, true),
        )
        .expect("from_raw");
        assert_eq!(a.subarray_order(), order, "the declared order must round-trip");
        let back = a.to_raw().expect("to_raw");
        for p in 0..raw.len() {
            assert_eq!(back[p].to_bits_pair(), raw[p].to_bits_pair(), "order {order:?} elem {p}");
        }
    }
    // A buffer of the wrong length is refused.
    assert!(
        KsymmArray::from_raw(
            &raw[..raw.len() - 1],
            &[NOCC, NVIR],
            SubarrayOrder::C,
            meta_2d(kpts, &rmat, &label, &trans, true)
        )
        .is_err()
    );
}

#[test]
fn setitem_at_a_non_irreducible_key_writes_the_representative_and_reads_back_transformed() {
    let s = sym();
    let kpts = &s.kpts;
    let rmat = synthetic_rmat(kpts.nkpts(), n_ops(kpts), NOCC, NVIR, 7);
    let label = parse_label("oo", 2).expect("l");
    let trans = parse_trans("cn", 2).expect("t");
    let bl = NOCC * NOCC;

    let mut a = KsymmArray::zeros(
        &[NOCC, NOCC],
        SubarrayOrder::C,
        meta_2d(kpts, &rmat, &label, &trans, true),
    )
    .expect("zeros");

    // Fill every IBZ representative.
    let mut rng = Rng::new(0x5E7);
    let blocks: Vec<Vec<Complex64>> =
        kpts.ibz2bz.iter().map(|_| (0..bl).map(|_| rng.next_c()).collect()).collect();
    let keys = kpts.ibz2bz.clone();
    let vals: Vec<&[Complex64]> = blocks.iter().map(|b| b.as_slice()).collect();
    a.set_2d_many(&keys, &vals).expect("set");

    // Writing at a NON-representative key is silently discarded (upstream's
    // warning branch, `ktensor.py:245-247`) — the store must be unchanged.
    let ki = a_non_representative_k(kpts);
    let junk = vec![Complex64::new(1e9, -1e9); bl];
    let snapshot: Vec<Vec<Complex64>> =
        (0..a.n_blocks()).map(|m| a.stored_block(m).expect("s")).collect();
    a.set_2d_at(ki, &junk).expect("set at a reducible key must not error");
    for m in 0..a.n_blocks() {
        assert_eq!(a.stored_block(m).expect("s"), snapshot[m], "slot {m} must be untouched");
    }

    // Reading back at that key gives the TRANSFORM of the representative,
    // not the representative itself.
    let got = a.get_2d(ki).expect("get");
    let ki_ibz = kpts.bz2ibz[ki];
    let rep = a.stored_block(ki_ibz).expect("rep");
    let iop = kpts.stars_ops_bz[ki];
    let ri = ref_conj(rot_ref(&rmat, OrbSpace::Occ, kpts.ibz2bz[ki_ibz], iop), Conj::C);
    let rj = ref_conj(rot_ref(&rmat, OrbSpace::Occ, kpts.ibz2bz[ki_ibz], iop), Conj::N);
    let want = ref_2d(&rep, NOCC, NOCC, &ri, &rj);
    let mut w = Worst::new();
    for p in 0..bl {
        w.see((got[p] - want[p]).norm(), || format!("elem {p}"));
    }
    w.report("|getitem at a reducible key - independent einsum|", EINSUM_TOL);
    // ... and it is NOT simply the stored block.
    let differs = (0..bl).map(|p| (got[p] - rep[p]).norm()).fold(0.0_f64, f64::max);
    assert!(differs > 1e-6, "a reducible key must NOT read back the raw representative");

    // Reading back AT the representative is the stored block, bit-exactly.
    let at_rep = a.get_2d(kpts.ibz2bz[ki_ibz]).expect("get rep");
    for p in 0..bl {
        assert_eq!(at_rep[p].to_bits_pair(), rep[p].to_bits_pair());
    }
}

#[test]
fn unfolding_the_whole_ibz_and_refolding_reproduces_the_stored_blocks_bit_exactly() {
    let s = sym();
    let (kpts, kqrts) = (&s.kpts, &s.kqrts);
    let rmat = synthetic_rmat(kpts.nkpts(), n_ops(kpts), NOCC, NVIR, 8);

    // --- rank 2 ---------------------------------------------------------
    let label2 = parse_label("oo", 2).expect("l");
    let trans2 = parse_trans("cn", 2).expect("t");
    let bl2 = NOCC * NOCC;
    let mut rng = Rng::new(0xF01D);
    let raw2: Vec<Complex64> = (0..kpts.nkpts_ibz() * bl2).map(|_| rng.next_c()).collect();
    let a = KsymmArray::from_raw(
        &raw2,
        &[NOCC, NOCC],
        SubarrayOrder::C,
        meta_2d(kpts, &rmat, &label2, &trans2, true),
    )
    .expect("from_raw");
    let all: Vec<usize> = (0..kpts.nkpts()).collect();
    let unfolded = a.get_2d_many(&all).expect("unfold");
    let vals: Vec<&[Complex64]> = unfolded.iter().map(|b| b.as_slice()).collect();
    let mut refolded = KsymmArray::zeros(
        &[NOCC, NOCC],
        SubarrayOrder::C,
        meta_2d(kpts, &rmat, &label2, &trans2, true),
    )
    .expect("zeros");
    refolded.set_2d_many(&all, &vals).expect("refold");
    for m in 0..a.n_blocks() {
        let x = a.stored_block(m).expect("x");
        let y = refolded.stored_block(m).expect("y");
        for p in 0..bl2 {
            assert_eq!(x[p].to_bits_pair(), y[p].to_bits_pair(), "2d refold slot {m} elem {p}");
        }
    }

    // --- rank 4 ---------------------------------------------------------
    let label4 = parse_label("oovv", 4).expect("l");
    let trans4 = parse_trans("nncc", 4).expect("t");
    let d = [NOCC, NOCC, NVIR, NVIR];
    let bl4: usize = d.iter().product();
    let raw4: Vec<Complex64> =
        (0..kqrts.kqrts_ibz.len() * bl4).map(|_| rng.next_c()).collect();
    let b = KsymmArray::from_raw(
        &raw4,
        &d,
        SubarrayOrder::C,
        meta_4d(kpts, kqrts, &rmat, &label4, &trans4, true),
    )
    .expect("from_raw");
    let n = kpts.nkpts();
    let mut triples = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for aa in 0..n {
                triples.push([i, j, aa]);
            }
        }
    }
    let unfolded4 = b.get_4d_many(&triples).expect("unfold 4d");
    let vals4: Vec<&[Complex64]> = unfolded4.iter().map(|x| x.as_slice()).collect();
    let mut refolded4 = KsymmArray::zeros(
        &d,
        SubarrayOrder::C,
        meta_4d(kpts, kqrts, &rmat, &label4, &trans4, true),
    )
    .expect("zeros");
    refolded4.set_4d_many(&triples, &vals4).expect("refold");
    for m in 0..b.n_blocks() {
        let x = b.stored_block(m).expect("x");
        let y = refolded4.stored_block(m).expect("y");
        for p in 0..bl4 {
            assert_eq!(x[p].to_bits_pair(), y[p].to_bits_pair(), "4d refold slot {m} elem {p}");
        }
    }
}

#[test]
fn incore_and_outcore_give_identical_results() {
    let s = sym();
    let (kpts, kqrts) = (&s.kpts, &s.kqrts);
    let rmat = synthetic_rmat(kpts.nkpts(), n_ops(kpts), NOCC, NVIR, 9);
    let label = parse_label("oovv", 4).expect("l");
    let trans = parse_trans("ccnn", 4).expect("t");
    let d = [NOCC, NOCC, NVIR, NVIR];
    let bl: usize = d.iter().product();
    let mut rng = Rng::new(0x0C0);
    let raw: Vec<Complex64> = (0..kqrts.kqrts_ibz.len() * bl).map(|_| rng.next_c()).collect();

    let a_in = KsymmArray::from_raw(
        &raw,
        &d,
        SubarrayOrder::C,
        meta_4d(kpts, kqrts, &rmat, &label, &trans, true),
    )
    .expect("incore");
    let a_out = KsymmArray::from_raw(
        &raw,
        &d,
        SubarrayOrder::C,
        meta_4d(kpts, kqrts, &rmat, &label, &trans, false),
    )
    .expect("outcore");
    assert!(a_in.is_incore());
    assert!(!a_out.is_incore(), "the outcore branch must actually be out of core");

    let n = kpts.nkpts();
    let mut triples = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for aa in 0..n {
                triples.push([i, j, aa]);
            }
        }
    }
    let x = a_in.get_4d_many(&triples).expect("incore unfold");
    let y = a_out.get_4d_many(&triples).expect("outcore unfold");
    assert_eq!(x.len(), y.len());
    for t in 0..x.len() {
        for p in 0..bl {
            assert_eq!(
                x[t][p].to_bits_pair(),
                y[t][p].to_bits_pair(),
                "incore/outcore differ at triple {:?} elem {p}",
                triples[t]
            );
        }
    }
    // to_dense too.
    let dx = a_in.to_dense().expect("dense in");
    let dy = a_out.to_dense().expect("dense out");
    assert_eq!(dx.len(), dy.len());
    for p in 0..dx.len() {
        assert_eq!(dx[p].to_bits_pair(), dy[p].to_bits_pair(), "to_dense differs at {p}");
    }

    // And a WRITE through the out-of-core store lands where the incore one does.
    let mut b_out = KsymmArray::zeros(
        &d,
        SubarrayOrder::C,
        meta_4d(kpts, kqrts, &rmat, &label, &trans, false),
    )
    .expect("zeros outcore");
    let blocks: Vec<Vec<Complex64>> = (0..triples.len())
        .map(|t| dense_block_4d(triples[t][0], triples[t][1], triples[t][2], bl))
        .collect();
    let vals: Vec<&[Complex64]> = blocks.iter().map(|v| v.as_slice()).collect();
    b_out.set_4d_many(&triples, &vals).expect("outcore set");
    for (m, q) in kqrts.kqrts_ibz.iter().enumerate() {
        assert_eq!(
            b_out.stored_block(m).expect("s"),
            dense_block_4d(q[0], q[1], q[2], bl),
            "outcore slot {m}"
        );
    }
}

#[test]
fn the_unfold_is_bit_identical_at_1_and_8_rayon_workers() {
    let s = sym();
    let (kpts, kqrts) = (&s.kpts, &s.kqrts);
    let rmat = synthetic_rmat(kpts.nkpts(), n_ops(kpts), NOCC, NVIR, 10);
    let label = parse_label("oovv", 4).expect("l");
    let trans = parse_trans("nncc", 4).expect("t");
    let d = [NOCC, NOCC, NVIR, NVIR];
    let bl: usize = d.iter().product();
    let mut rng = Rng::new(0x8A0);
    let raw: Vec<Complex64> = (0..kqrts.kqrts_ibz.len() * bl).map(|_| rng.next_c()).collect();
    let a = KsymmArray::from_raw(
        &raw,
        &d,
        SubarrayOrder::C,
        meta_4d(kpts, kqrts, &rmat, &label, &trans, true),
    )
    .expect("from_raw");

    let n = kpts.nkpts();
    let mut triples = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for aa in 0..n {
                triples.push([i, j, aa]);
            }
        }
    }

    let run = |workers: usize| -> Vec<Vec<Complex64>> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .expect("thread pool");
        pool.install(|| a.get_4d_many(&triples).expect("unfold"))
    };
    let one = run(1);
    let eight = run(8);
    assert_eq!(one.len(), eight.len());
    for t in 0..one.len() {
        for p in 0..bl {
            assert_eq!(
                one[t][p].to_bits_pair(),
                eight[t][p].to_bits_pair(),
                "1-vs-8 worker mismatch at triple {:?} elem {p}",
                triples[t]
            );
        }
    }
    println!("  unfold: {} triples bit-identical at 1 and 8 workers", triples.len());
}

#[test]
fn missing_metadata_is_refused_not_guessed() {
    let s = sym();
    let kpts = &s.kpts;
    let kqrts = &s.kqrts;
    let rmat = synthetic_rmat(kpts.nkpts(), n_ops(kpts), NOCC, NVIR, 11);
    let label = parse_label("oo", 2).expect("l");
    let trans = parse_trans("cn", 2).expect("t");

    // A rank-4 array without `kqrts`.
    let bare = KsymmMeta::new(kpts);
    assert!(KsymmArray::empty(&[NOCC, NOCC, NVIR, NVIR], SubarrayOrder::C, bare).is_err());

    // A rank-2 array without `rmat` builds, stores and reads back its own
    // stored blocks, but cannot TRANSFORM.
    let a = KsymmArray::empty(&[NOCC, NOCC], SubarrayOrder::C, bare).expect("empty");
    assert!(a.get_2d(0).is_err(), "no rmat/label/trans -> transform must refuse");

    // Wrong rank for the accessor.
    let m4 = meta_4d(kpts, kqrts, &rmat, &label, &trans, true);
    let _ = m4; // metadata with a rank-2 label is only rejected on use
    let a2 = KsymmArray::empty(
        &[NOCC, NOCC],
        SubarrayOrder::C,
        meta_2d(kpts, &rmat, &label, &trans, true),
    )
    .expect("empty");
    assert!(a2.get_4d([0, 0, 0]).is_err(), "rank-2 array must refuse a 4-d getter");
}

/// D-17-06-01, made falsifiable on this fixture rather than argued on paper.
///
/// Upstream's `fromdense` rank-2 branch (`ktensor.py:194-198`) computes
/// `ki_ibz = kpts.bz2ibz[ki]` and then passes `ki_ibz` — an IBZ index — to
/// `__setitem__`, which `set_2d` (`:244-250`) treats as a FULL-BZ index.
/// On `si [2,2,2]` the IBZ is `ibz2bz = [0, 6, 7]`, so the IBZ indices
/// `{0, 1, 2}` intersect it in `{0}` alone: upstream would keep ONE of the
/// three blocks and warn away the other two. This test pins that arithmetic,
/// so the deviation cannot be "simplified back" to upstream's line.
#[test]
fn upstreams_fromdense_key_choice_would_drop_two_of_three_blocks_here() {
    let s = sym();
    let kpts = &s.kpts;
    assert_eq!(kpts.ibz2bz, vec![0, 6, 7], "the fixture's IBZ must be [0, 6, 7]");
    let ibz_indices: Vec<usize> = (0..kpts.nkpts_ibz()).collect();
    let survivors: Vec<usize> =
        ibz_indices.iter().copied().filter(|k| kpts.ibz2bz.contains(k)).collect();
    println!(
        "  upstream fromdense would pass keys {ibz_indices:?} to set_2d; \
         only {survivors:?} are in ibz2bz = {:?}",
        kpts.ibz2bz
    );
    assert_eq!(survivors, vec![0], "upstream's key choice keeps only slot 0 here");

    // The port's key choice (BZ indices) keeps all three.
    let kept: Vec<usize> =
        kpts.ibz2bz.iter().copied().filter(|k| kpts.ibz2bz.contains(k)).collect();
    assert_eq!(kept, kpts.ibz2bz, "the ported key choice keeps every representative");
}

// =====================================================================
// Task 1 speed — the incore / out-of-core crossover, MEASURED
// =====================================================================

/// Upstream has NO size constant to copy: `incore` comes straight from
/// `metadata.get('incore', True)` (`ktensor.py:50`) and every caller decides
/// with a pure MEMORY test — `_memory_4d(...) + lib.current_memory()[0] <
/// cc.max_memory * .9` (`kccsd_rhf_ksymm.py:153-166`, `:218-235`,
/// `:289-323`, `:413-417`). So the question this measurement answers is not
/// "where is upstream's threshold" but "what does forcing the out-of-core
/// path COST when the tensor would have fit", which is what decides whether
/// the Rust port needs a minimum-size floor on top of the memory test.
///
/// Prints a table under `--nocapture`; asserts only that both paths agree.
#[test]
fn measure_the_incore_outcore_crossover() {
    let s = sym();
    let (kpts, kqrts) = (&s.kpts, &s.kqrts);
    let n = kpts.nkpts();
    let nblk = kqrts.kqrts_ibz.len();

    let mut triples = Vec::new();
    for i in 0..n {
        for j in 0..n {
            for aa in 0..n {
                triples.push([i, j, aa]);
            }
        }
    }

    println!();
    println!(
        "  incore/out-of-core crossover — si [2,2,2], nkpts = {n}, stored 4-d blocks = {nblk}"
    );
    println!(
        "  {:>6} {:>12} {:>11} {:>11} {:>8} {:>11} {:>11} {:>8}",
        "nocc/", "stored", "from_raw", "from_raw", "ratio", "unfold", "unfold", "ratio"
    );
    println!(
        "  {:>6} {:>12} {:>11} {:>11} {:>8} {:>11} {:>11} {:>8}",
        "nvir", "MiB", "incore", "outcore", "", "incore", "outcore", ""
    );

    for &(nocc, nvir) in &[(1usize, 1usize), (2, 2), (3, 4), (4, 6), (6, 8), (8, 10), (10, 14)] {
        let rmat = synthetic_rmat(n, n_ops(kpts), nocc, nvir, 99);
        let label = vec![OrbSpace::Occ, OrbSpace::Occ, OrbSpace::Vir, OrbSpace::Vir];
        let trans = vec![Conj::N, Conj::N, Conj::C, Conj::C];
        let d = [nocc, nocc, nvir, nvir];
        let bl: usize = d.iter().product();
        let mut rng = Rng::new(0x3A50);
        let raw: Vec<Complex64> = (0..nblk * bl).map(|_| rng.next_c()).collect();
        let mib = (nblk * bl * 16) as f64 / (1024.0 * 1024.0);

        let t0 = std::time::Instant::now();
        let a_in = KsymmArray::from_raw(
            &raw,
            &d,
            SubarrayOrder::C,
            meta_4d(kpts, kqrts, &rmat, &label, &trans, true),
        )
        .expect("incore");
        let t_build_in = t0.elapsed().as_secs_f64();

        let t0 = std::time::Instant::now();
        let a_out = KsymmArray::from_raw(
            &raw,
            &d,
            SubarrayOrder::C,
            meta_4d(kpts, kqrts, &rmat, &label, &trans, false),
        )
        .expect("outcore");
        let t_build_out = t0.elapsed().as_secs_f64();

        let t0 = std::time::Instant::now();
        let x = a_in.get_4d_many(&triples).expect("unfold in");
        let t_unfold_in = t0.elapsed().as_secs_f64();

        let t0 = std::time::Instant::now();
        let y = a_out.get_4d_many(&triples).expect("unfold out");
        let t_unfold_out = t0.elapsed().as_secs_f64();

        for t in 0..x.len() {
            for p in 0..bl {
                assert_eq!(x[t][p].to_bits_pair(), y[t][p].to_bits_pair());
            }
        }

        println!(
            "  {:>6} {:>12.4} {:>10.4}s {:>10.4}s {:>8.1}x {:>10.4}s {:>10.4}s {:>8.2}x",
            format!("{nocc}/{nvir}"),
            mib,
            t_build_in,
            t_build_out,
            t_build_out / t_build_in.max(1e-9),
            t_unfold_in,
            t_unfold_out,
            t_unfold_out / t_unfold_in.max(1e-9),
        );
    }
    println!();
}

// =====================================================================
// Task 4 — the acceptance test uses a REAL converged SCF, not a synthetic
// rotation
// =====================================================================
//
// 17-06-PLAN.md Task 4 asks for "a real `khf_ksymm` Fock store". 17-07
// (`khf_ksymm`) has NOT shipped yet — it is the next plan — so this file
// uses the quantity that DOES exist today and that upstream's own
// `KsymmArray` callers store: the MO-basis one-electron blocks.
//
//   `kintermediates_rhf_ksymm.py:26-33`  Fki = ktensor.empty([nocc,nocc],
//        metadata = {'label': 'oo', 'trans': 'cn'}),
//        then `Fki[ki] = eris.fock[ki,:nocc,:nocc]`
//   `:47-56`  Fkc, label 'ov', trans 'cn', `eris.fock[ki,:nocc,nocc:]`
//   `:71-80`  Fac, label 'vv', trans 'cn', `eris.fock[ki,nocc:,nocc:]`
//
// So `('oo'|'ov'|'vv', 'cn')` applied to `C^H O C` is upstream's own
// contract, and the transform law it encodes is
//
//   A[k2] = rot^H A[k1] rot,   rot = C[k1]^H S[k1] R^H C[k2]
//
// which holds for any operator `O` commuting with the space group. The
// converged Fock is one; `hcore` is another, and `hcore` is the STRONGER
// probe because `C^H F C` is diagonal in the canonical MO basis while
// `C^H h C` is not — a diagonal reference cannot see a wrong mixing inside
// a degenerate subspace. Both are measured.
//
// The opposite `trans` ('nc') is measured too and asserted LARGE, so the
// combination cannot be flipped without deleting an assertion.

/// 17-01 Task 2's Gate-B floor: at PySCF's default `cell.precision = 1e-8`
/// the transform residual is 4.481e-10, and it is `cell.precision`-limited,
/// not `conv_tol`-limited. This fixture runs at `precision = 1e-10` /
/// `conv_tol_grad = 1e-10`, which 17-05 measured at 1.784e-11 for
/// `transform_dm`. Gated at 1e-9 — 17-01's floor — with the measured
/// maximum printed under `--nocapture`. **If it fails, tighten the fixture,
/// do not relax the gate.**
const GATE_B_TOL: f64 = 1e-9;
const FIXTURE_PRECISION: f64 = 1e-10;
const FIXTURE_CONV_TOL_GRAD: f64 = 1e-10;

struct RealFixture {
    kpts: KPoints,
    nocc: usize,
    nvir: usize,
    rmat: MORotationMatrix,
    /// `C_o^H F C_o` per BZ k, row-major `nocc x nocc`.
    fock_oo: Vec<Vec<Complex64>>,
    /// `C_o^H h C_o` per BZ k, row-major `nocc x nocc`.
    hcore_oo: Vec<Vec<Complex64>>,
    /// `C_o^H h C_v` per BZ k, row-major `nocc x nvir`.
    hcore_ov: Vec<Vec<Complex64>>,
    /// `C_v^H h C_v` per BZ k, row-major `nvir x nvir`.
    hcore_vv: Vec<Vec<Complex64>>,
}

/// F-order (column-major) `n x n` `CTensor` -> row-major.
fn f_square_to_rowmajor(ct: &pyscf_algebra::CTensor, n: usize) -> Vec<Complex64> {
    let mut out = vec![C0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = Complex64::new(ct.re[i + j * n], ct.im[i + j * n]);
        }
    }
    out
}

fn rowmajor_square(ct: &pyscf_algebra::CTensor, n: usize) -> Vec<Complex64> {
    (0..n * n).map(|k| Complex64::new(ct.re[k], ct.im[k])).collect()
}

fn colmajor_rect_to_rowmajor(
    ct: &pyscf_algebra::CTensor,
    nrows: usize,
    ncols: usize,
) -> Vec<Complex64> {
    let mut out = vec![C0; nrows * ncols];
    for r in 0..nrows {
        for c in 0..ncols {
            out[r * ncols + c] = Complex64::new(ct.re[r + c * nrows], ct.im[r + c * nrows]);
        }
    }
    out
}

/// `mo[:, lo:hi]` of a row-major `nao x nmo` matrix.
fn cols(mo: &[Complex64], nao: usize, nmo: usize, lo: usize, hi: usize) -> Vec<Complex64> {
    let w = hi - lo;
    let mut out = vec![C0; nao * w];
    for r in 0..nao {
        for c in lo..hi {
            out[r * w + (c - lo)] = mo[r * nmo + c];
        }
    }
    out
}

/// `C_l^H O C_r` for row-major `nao x nl`, `nao x nao`, `nao x nr`.
fn project(cl: &[Complex64], nl: usize, op: &[Complex64], cr: &[Complex64], nr: usize, nao: usize)
-> Vec<Complex64> {
    // t = O @ C_r  (nao x nr)
    let mut t = vec![C0; nao * nr];
    for i in 0..nao {
        for p in 0..nao {
            let a = op[i * nao + p];
            if a == C0 {
                continue;
            }
            for j in 0..nr {
                t[i * nr + j] += a * cr[p * nr + j];
            }
        }
    }
    // out = C_l^H @ t  (nl x nr)
    let mut out = vec![C0; nl * nr];
    for i in 0..nl {
        for p in 0..nao {
            let a = cl[p * nl + i].conj();
            if a == C0 {
                continue;
            }
            for j in 0..nr {
                out[i * nr + j] += a * t[p * nr + j];
            }
        }
    }
    out
}

fn real_fixture() -> &'static RealFixture {
    use pyscf_pbc_df::JkOpts;
    use pyscf_pbc_scf::krhf::to_row_major;
    use pyscf_pbc_scf::{KInitGuess, KScfConfig, KScfResult, Krhf};

    static G: std::sync::OnceLock<RealFixture> = std::sync::OnceLock::new();
    G.get_or_init(|| {
        let mut cell = si_precision(FIXTURE_PRECISION);
        cell.space_group_symmetry = true;
        cell.symmorphic = false;
        let check_mesh_symmetry = !cell._mesh_from_build;
        build_lattice_symmetry(&mut cell, check_mesh_symmetry).expect("build_lattice_symmetry");
        let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
        let kpts = make_kpts(&cell, &kpts_abs, true, true).expect("make_kpts");
        assert!(!kpts.time_reversal, "si has inversion: time reversal must be OFF");

        let mf = Krhf::new(cell.clone(), &kpts_abs).expect("Krhf::new");
        let cfg = KScfConfig {
            conv_tol: 1e-11,
            conv_tol_grad: Some(FIXTURE_CONV_TOL_GRAD),
            max_cycle: 50,
            init_guess: KInitGuess::Minao,
            ..KScfConfig::default()
        };
        let r: KScfResult = mf.kernel(&cfg).expect("full-BZ KRHF must run");
        assert!(r.converged, "full-BZ KRHF did not converge in {} cycles", r.cycles);

        let nao = cell.mol.nao_nr;
        let nkpts = kpts_abs.len();
        let nmo = r.mo_occ[0].len();
        let nocc = r.mo_occ[0].iter().filter(|&&o| o > 0.0).count();
        let nvir = nmo - nocc;

        // hcore, row-major, and the converged Fock (khf.py:670-695 minus eig).
        let hcore = to_row_major(
            pyscf_pbc_df::get_hcore(mf.with_df.as_ref(), mf.kpts()).expect("get_hcore"),
            nao,
        );
        let mut fock_ct = hcore.clone();
        let jk = mf
            .with_df
            .get_jk(
                &r.dm,
                mf.kpts(),
                JkOpts {
                    hermi: 1,
                    kpts_band: None,
                    with_j: true,
                    with_k: true,
                    exxdiv: mf.exxdiv,
                    omega: None,
                    kk_symmetry: false,
                },
            )
            .expect("get_jk");
        let vj = jk.vj.expect("vj");
        let vk = jk.vk.expect("vk");
        for (k, f) in fock_ct.iter_mut().enumerate() {
            for i in 0..f.re.len() {
                f.re[i] += vj[0][k].re[i] - 0.5 * vk[0][k].re[i];
                f.im[i] += vj[0][k].im[i] - 0.5 * vk[0][k].im[i];
            }
        }

        let ovlp: Vec<Vec<Complex64>> =
            pyscf_pbc_gto::hcore::get_ovlp(&cell, &kpts_abs)
                .expect("get_ovlp")
                .iter()
                .map(|s| f_square_to_rowmajor(s, nao))
                .collect();
        let mo_coeff: Vec<Vec<Complex64>> = (0..nkpts)
            .map(|k| colmajor_rect_to_rowmajor(&r.mo_coeff[r.idx(0, k)], nao, nmo))
            .collect();
        let hcore_rm: Vec<Vec<Complex64>> =
            (0..nkpts).map(|k| rowmajor_square(&hcore[k], nao)).collect();
        let fock_rm: Vec<Vec<Complex64>> =
            (0..nkpts).map(|k| rowmajor_square(&fock_ct[k], nao)).collect();

        let mut rmat = MORotationMatrix::new(nocc, nmo);
        rmat.build(&kpts, &cell, &mo_coeff, &ovlp, nao).expect("MORotationMatrix::build");

        let mut fock_oo = Vec::with_capacity(nkpts);
        let mut hcore_oo = Vec::with_capacity(nkpts);
        let mut hcore_ov = Vec::with_capacity(nkpts);
        let mut hcore_vv = Vec::with_capacity(nkpts);
        for k in 0..nkpts {
            let co = cols(&mo_coeff[k], nao, nmo, 0, nocc);
            let cv = cols(&mo_coeff[k], nao, nmo, nocc, nmo);
            fock_oo.push(project(&co, nocc, &fock_rm[k], &co, nocc, nao));
            hcore_oo.push(project(&co, nocc, &hcore_rm[k], &co, nocc, nao));
            hcore_ov.push(project(&co, nocc, &hcore_rm[k], &cv, nvir, nao));
            hcore_vv.push(project(&cv, nvir, &hcore_rm[k], &cv, nvir, nao));
        }

        RealFixture { kpts, nocc, nvir, rmat, fock_oo, hcore_oo, hcore_ov, hcore_vv }
    })
}

/// Store a real MO-basis quantity over the IBZ, read back EVERY BZ
/// k-point, compare against the dense full-BZ array. Returns the maximum
/// residual for `trans` and for the opposite `trans` (the control).
fn acceptance_round(
    g: &'static RealFixture,
    dense: &[Vec<Complex64>],
    label_s: &str,
    trans_s: &str,
    opposite_s: &str,
    di: usize,
    dj: usize,
) -> (f64, f64) {
    let label = parse_label(label_s, 2).expect("label");
    let mut maxima = [0.0f64; 2];
    for (slot, t) in [trans_s, opposite_s].iter().enumerate() {
        let trans = parse_trans(t, 2).expect("trans");
        let flat: Vec<Complex64> = dense.iter().flat_map(|b| b.iter().copied()).collect();
        let a = KsymmArray::from_dense(
            &flat,
            &[di, dj],
            SubarrayOrder::C,
            meta_2d(&g.kpts, &g.rmat, &label, &trans, true),
        )
        .expect("from_dense");
        let back = a.to_dense().expect("to_dense");
        let mut worst = Worst::new();
        for k in 0..g.kpts.nkpts() {
            for p in 0..di * dj {
                worst.see((back[k * di * dj + p] - dense[k][p]).norm(), || {
                    format!("k = {k}, elem {p}")
                });
            }
        }
        println!(
            "  acceptance label '{label_s}' trans '{t}': max |KsymmArray - dense| = {:e}  (at {})",
            worst.val, worst.at
        );
        maxima[slot] = worst.val;
    }
    (maxima[0], maxima[1])
}

/// `(max |off-diagonal|, max |Im|)` over every k of a square MO block.
fn block_structure(dense: &[Vec<Complex64>], n: usize) -> (f64, f64) {
    let mut off: f64 = 0.0;
    let mut im: f64 = 0.0;
    for a in dense {
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off = off.max(a[i * n + j].norm());
                }
                im = im.max(a[i * n + j].im.abs());
            }
        }
    }
    (off, im)
}

/// `max |Im|` over every k of a rectangular MO block.
fn max_imag(dense: &[Vec<Complex64>]) -> f64 {
    dense.iter().flat_map(|a| a.iter()).fold(0.0f64, |m, z| m.max(z.im.abs()))
}

#[test]
fn acceptance_real_converged_krhf_mo_blocks_round_trip_through_ksymmarray() {
    let g = real_fixture();
    println!(
        "  fixture: si [2,2,2] KRHF/FFTDF, precision {FIXTURE_PRECISION:e}, \
         conv_tol_grad {FIXTURE_CONV_TOL_GRAD:e}; nocc = {}, nvir = {}, \
         nkpts = {}, nkpts_ibz = {}",
        g.nocc,
        g.nvir,
        g.kpts.nkpts(),
        g.kpts.nkpts_ibz()
    );

    // Structure of the reference blocks, over EVERY k — this is what decides
    // whether the fixture can separate `trans = 'cn'` from `'nc'` at all:
    //   'cn' -> R^H A R,   'nc' -> R^T A R*.
    // For a REAL `A` the second is the conjugate of the first, and if the
    // result is also real the two coincide identically. Measured, not assumed.
    let (foff, fim) = block_structure(&g.fock_oo, g.nocc);
    let (hoff, him) = block_structure(&g.hcore_oo, g.nocc);
    let (voff, vim) = block_structure(&g.hcore_vv, g.nvir);
    let ovim = max_imag(&g.hcore_ov);
    println!("  block structure (max over all {} k-points):", g.kpts.nkpts());
    println!("    C_o^H F C_o : max|off-diag| = {foff:e}   max|Im| = {fim:e}");
    println!("    C_o^H h C_o : max|off-diag| = {hoff:e}   max|Im| = {him:e}");
    println!("    C_v^H h C_v : max|off-diag| = {voff:e}   max|Im| = {vim:e}");
    println!("    C_o^H h C_v :                              max|Im| = {ovim:e}");

    // (label, dense, di, dj, name)
    let cases: [(&str, &Vec<Vec<Complex64>>, usize, usize, &str); 4] = [
        ("oo", &g.fock_oo, g.nocc, g.nocc,
         "C_o^H F C_o  (upstream's eris.fock oo block, kintermediates_rhf_ksymm.py:26-33)"),
        ("oo", &g.hcore_oo, g.nocc, g.nocc, "C_o^H h C_o"),
        ("ov", &g.hcore_ov, g.nocc, g.nvir, "C_o^H h C_v"),
        ("vv", &g.hcore_vv, g.nvir, g.nvir, "C_v^H h C_v"),
    ];

    let mut overall = 0.0f64;
    let mut best_control = 0.0f64;
    for (label, dense, di, dj, name) in cases.iter() {
        println!("  --- {name} ---");
        let (good, control) = acceptance_round(g, dense, label, "cn", "nc", *di, *dj);
        assert!(
            good < GATE_B_TOL,
            "label '{label}' trans 'cn' — upstream's own combination for a \
             one-electron MO block (kintermediates_rhf_ksymm.py:26-33, :47-56, \
             :71-80) — gave {good:e}, above the 17-01 Gate-B floor \
             {GATE_B_TOL:e}. Tighten the fixture, do NOT relax the gate."
        );
        overall = overall.max(good);
        best_control = best_control.max(control);
    }
    println!("  acceptance: worst 'cn' residual over all four real quantities = {overall:e}");
    println!("  acceptance: best  'nc' (wrong-trans) residual                 = {best_control:e}");
    // See the module-level note and 17-06-SUMMARY: whether this fixture can
    // discriminate 'cn' from 'nc' is a property of the DATA, not of the port,
    // and is recorded rather than assumed. The assertion below is written
    // from the measured value.
    assert!(
        best_control > 1e-6 || (fim + him + vim + ovim) < 1e-9,
        "the wrong `trans` was indistinguishable ({best_control:e}) even though \
         the reference blocks carry an imaginary part \
         (max|Im| = {:e}); that combination is supposed to differ, so either \
         the fixture or `transform_2d` changed",
        fim.max(him).max(vim).max(ovim)
    );
}

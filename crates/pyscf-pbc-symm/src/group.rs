//! Port of `pyscf/pbc/symm/group.py` (476 l) — crystallographic point-group
//! elements (`PGElement`), finite-group algebra (`FiniteGroup`,
//! `PointGroup`) and representation bookkeeping (`Representation`).
//!
//! `PGElement.__hash__`/`decrypt_hash` (`group.py:80-115`) encode a `3x3`
//! integer rotation matrix into a single integer with the identity moved to
//! the origin. That encoding IS the group's element ordering
//! (`__lt__`, `:117-120`), and `character_table`'s row order is read
//! POSITIONALLY by `basis.py` downstream — porting a different-but-valid
//! encoding would silently reorder rows there. See [`PgElement::hash_key`].
//!
//! `character_table` (`group.py:313-360`) is Burnside's method over the
//! class algebra: a random linear combination of class sums, diagonalised.
//! It needs a general (non-symmetric) real eigendecomposition, which
//! `pyscf-algebra` does not expose (only the symmetric `eigh_gen`, ALG-05).
//! This crate never touches cubecl or a device tensor, so pulling `faer`'s
//! host-only `Eigen::new_from_real` directly here is not an ALG-06 wall
//! violation (see the `Cargo.toml` comment). The "random" weights use a
//! FIXED-seed `SplitMix64` rather than upstream's `np.random.rand`, so this
//! port's own tests are reproducible; the random draw only exists to break
//! ties among degenerate eigenvalues of the class algebra generically, and
//! every identity this plan gates (Latin-square multiplication table,
//! Burnside orthogonality, `chi_to_rep(rep_to_chi(r)) == r`) holds for ANY
//! generic draw, not upstream's specific one.

use std::collections::HashMap;

use num_complex::Complex;

use crate::error::PbcSymmError;
use crate::geom::RotMatrix;
use crate::tables;

/// `num_complex::Complex<f64>` — same underlying type as `faer::c64`.
pub type Complex64 = Complex<f64>;

/// `group.py:25-27` — `_round_zero`: zero out entries whose COMPLEX MODULUS
/// (not real/imag part separately) is below `tol`.
fn round_zero(chi: &mut [Vec<Complex64>], tol: f64) {
    for row in chi.iter_mut() {
        for c in row.iter_mut() {
            if c.norm() < tol {
                *c = Complex64::new(0.0, 0.0);
            }
        }
    }
}

fn round9(x: f64) -> f64 {
    (x * 1e9).round() / 1e9
}

/// Lexicographic ordering numpy uses for complex arrays (and hence
/// `np.lexsort`): compare the real part, then the imaginary part.
fn complex_cmp(a: Complex64, b: Complex64) -> std::cmp::Ordering {
    a.re.total_cmp(&b.re).then_with(|| a.im.total_cmp(&b.im))
}

// ---------------------------------------------------------------------
// A minimal deterministic PRNG (splitmix64) — stands in for upstream's
// `np.random.rand`. See the module doc: any generic draw works here.
// ---------------------------------------------------------------------

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A pseudo-uniform draw in `[0, 1)`, 53 bits of randomness.
    fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        (bits as f64) / (1u64 << 53) as f64
    }
}

/// Fixed seed for [`FiniteGroup::character_table`]'s Burnside random weights
/// — reproducible across runs, see the module doc.
const CHARACTER_TABLE_SEED: u64 = 0x1727_5A5F_3B5F;

// ---------------------------------------------------------------------
// PGElement (group.py:56-133)
// ---------------------------------------------------------------------

/// Base-3 positional evaluation, most-significant digit first — shared by
/// `PGElement::hash_key` (`_id`, `group.py:81-83`) and [`decrypt_hash`]'s
/// `id_eye`/`id_max` constants.
fn base3_id(digits: &[i32]) -> i64 {
    let mut out: i64 = 0;
    for &d in digits {
        out = out * 3 + d as i64;
    }
    out
}

fn flatten_matrix(w: &RotMatrix) -> [i32; 9] {
    let mut out = [0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[3 * i + j] = w[i][j];
        }
    }
    out
}

/// `group.py:93-115` — `PGElement.decrypt_hash`, generalised over dimension
/// (`2` or `3`, matching upstream's two branches). Returns the flattened
/// (row-major) rotation matrix entries, values in `{-1, 0, 1}`.
pub fn decrypt_hash(h: i64, dimension: usize) -> Result<Vec<i32>, PbcSymmError> {
    let (id_eye, id_max) = match dimension {
        3 => (base3_id(&[2, 1, 1, 1, 2, 1, 1, 1, 2]), base3_id(&[2; 9])),
        2 => (base3_id(&[2, 1, 1, 2]), base3_id(&[2; 4])),
        _ => return Err(PbcSymmError::UnsupportedDimension(dimension)),
    };

    let mut r = h + id_eye;
    if r > id_max {
        r -= id_max + 1;
    }

    let ndigits = dimension * dimension;
    let mut digits = vec![0i32; ndigits];
    let mut num = r;
    for (place, d) in digits.iter_mut().enumerate() {
        let power = (ndigits - 1 - place) as u32;
        let base_pow = 3i64.pow(power);
        let ki = num / base_pow;
        num -= ki * base_pow;
        *d = ki as i32;
    }
    for d in digits.iter_mut() {
        *d -= 1;
    }
    Ok(digits)
}

/// `group.py:56-133` — `PGElement`: a crystallographic point-group element,
/// a `3x3` rotation matrix in the lattice-translation-vector basis. This
/// port only ever constructs 3-dimensional elements (`search_point_group_ops`
/// always returns `3x3` matrices, regardless of `cell.dimension` — see
/// `geom.rs`'s module doc), matching `PGElement.dimension = matrix.shape[0]`
/// being 3 in every path this phase exercises; [`decrypt_hash`] still ports
/// upstream's `dimension = 2` branch for parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PgElement {
    pub matrix: RotMatrix,
}

impl PgElement {
    pub fn new(matrix: RotMatrix) -> Self {
        Self { matrix }
    }

    /// `group.py:130` — `rot = matrix` (alias property).
    pub fn rot(&self) -> RotMatrix {
        self.matrix
    }

    /// `group.py:72-75` — `__matmul__`: `np.dot(self.matrix, other.matrix)`.
    pub fn compose(&self, other: &Self) -> Self {
        let mut out = [[0i32; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                let mut s = 0;
                for k in 0..3 {
                    s += self.matrix[i][k] * other.matrix[k][j];
                }
                *cell = s;
            }
        }
        Self::new(out)
    }

    /// `group.py:80-91` — `__hash__`.
    pub fn hash_key(&self) -> i64 {
        let flat = flatten_matrix(&self.matrix);
        let s: [i32; 9] = std::array::from_fn(|i| flat[i] + 1);
        let mut r = base3_id(&s);
        let id_eye = base3_id(&[2, 1, 1, 1, 2, 1, 1, 1, 2]);
        r -= id_eye;
        if r < 0 {
            let id_max = base3_id(&[2; 9]);
            r += id_max + 1;
        }
        r
    }

    /// `group.py:93-115`, specialised to `dimension = 3` (see the struct
    /// doc for why that is the only dimension this port needs).
    pub fn decrypt_hash3(h: i64) -> RotMatrix {
        // dimension = 3 is always Ok(_) — see `decrypt_hash`'s match arms.
        let digits = match decrypt_hash(h, 3) {
            Ok(d) => d,
            Err(_) => unreachable!("dimension = 3 is always supported"),
        };
        let mut m = [[0i32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = digits[3 * i + j];
            }
        }
        m
    }

    /// `group.py:132-133` — `inv`: `np.linalg.inv(self.matrix)` cast to
    /// `int32`. Ported as an EXACT integer adjugate-over-determinant instead
    /// of upstream's float-inverse-then-truncate: every `PGElement` this
    /// crate constructs is unimodular (`|det| == 1`, `search_point_group_ops`
    /// only admits metric-preserving integer matrices), so the two are
    /// mathematically identical, and the exact-integer route sidesteps a
    /// truncation trap upstream's own approach has (a value that should be
    /// exactly `1` but floating-inverts to `0.999999999998` truncates to
    /// `0`, not `1`, under `.astype(int32)`, which truncates toward zero
    /// rather than rounding).
    pub fn inv(&self) -> Self {
        let w = &self.matrix;
        let det = w[0][0] * (w[1][1] * w[2][2] - w[1][2] * w[2][1])
            - w[0][1] * (w[1][0] * w[2][2] - w[1][2] * w[2][0])
            + w[0][2] * (w[1][0] * w[2][1] - w[1][1] * w[2][0]);
        debug_assert!(det == 1 || det == -1, "PGElement must be unimodular");
        // adjugate = transpose of the cofactor matrix
        let cofactor = |r: usize, c: usize| -> i32 {
            let rows: Vec<usize> = (0..3).filter(|&i| i != r).collect();
            let cols: Vec<usize> = (0..3).filter(|&j| j != c).collect();
            let sign = if (r + c).is_multiple_of(2) { 1 } else { -1 };
            sign * (w[rows[0]][cols[0]] * w[rows[1]][cols[1]]
                - w[rows[0]][cols[1]] * w[rows[1]][cols[0]])
        };
        let mut inv = [[0i32; 3]; 3];
        for (i, row) in inv.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                // adjugate[i][j] = cofactor(j, i); inverse = adjugate / det.
                *cell = cofactor(j, i) / det;
            }
        }
        Self::new(inv)
    }
}

impl PartialOrd for PgElement {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// `group.py:117-120` — `__lt__`, via [`PgElement::hash_key`].
impl Ord for PgElement {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.hash_key().cmp(&other.hash_key())
    }
}

// ---------------------------------------------------------------------
// GroupElement trait + FiniteGroup (group.py:29-397)
// ---------------------------------------------------------------------

/// `group.py:29-53` — `GroupElement`, the abstract base every finite-group
/// element type implements.
pub trait GroupElement: Clone {
    fn compose(&self, other: &Self) -> Self;
    fn hash_key(&self) -> i64;
    fn inverse(&self) -> Self;
    /// `PointGroup.elements_from_hash` (`group.py:419-422`), pushed onto the
    /// element trait so [`FiniteGroup::from_hash`] is generic.
    fn from_hash_key(h: i64) -> Self;
}

impl GroupElement for PgElement {
    fn compose(&self, other: &Self) -> Self {
        PgElement::compose(self, other)
    }
    fn hash_key(&self) -> i64 {
        PgElement::hash_key(self)
    }
    fn inverse(&self) -> Self {
        PgElement::inv(self)
    }
    fn from_hash_key(h: i64) -> Self {
        PgElement::new(PgElement::decrypt_hash3(h))
    }
}

/// `group.py:136-396` — `FiniteGroup`.
#[derive(Debug, Clone)]
pub struct FiniteGroup<E> {
    pub elements: Vec<E>,
}

/// `pyscf.pbc.symm.symm.PointGroup` is `FiniteGroup` specialised to
/// `PGElement`; ported as a type alias plus an `impl` block below rather than
/// a wrapper struct, since Rust has no class inheritance to mirror
/// `class PointGroup(FiniteGroup)` directly.
pub type PointGroup = FiniteGroup<PgElement>;

impl<E: GroupElement> FiniteGroup<E> {
    /// `group.py:146-160` — `__init__` (the `from_hash=False` branch), plus
    /// `_check_sanity` (`:385-396`), run eagerly rather than lazily since
    /// this port does not defer construction validation.
    pub fn new(elements: Vec<E>) -> Result<Self, PbcSymmError> {
        let group = Self { elements };
        group.check_sanity()?;
        Ok(group)
    }

    /// `group.py:146-148` — `__init__(elements, from_hash=True)`, via
    /// `elements_from_hash`.
    pub fn from_hash(hashes: &[i64]) -> Result<Self, PbcSymmError> {
        let elements = hashes.iter().map(|&h| E::from_hash_key(h)).collect();
        Self::new(elements)
    }

    /// `group.py:167-168`, `202-206` — `__len__` / `order`.
    pub fn order(&self) -> usize {
        self.elements.len()
    }

    /// `group.py:212-219` — `hash_table`: `{hash(g): i}`.
    pub fn hash_table(&self) -> HashMap<i64, usize> {
        self.elements
            .iter()
            .enumerate()
            .map(|(i, g)| (g.hash_key(), i))
            .collect()
    }

    /// `group.py:225-236` — `inverse_table`.
    pub fn inverse_table(&self) -> Vec<usize> {
        let ht = self.hash_table();
        self.elements
            .iter()
            .map(|g| ht[&g.inverse().hash_key()])
            .collect()
    }

    /// `group.py:242-254` — `multiplication_table`.
    pub fn multiplication_table(&self) -> Vec<Vec<usize>> {
        let ht = self.hash_table();
        let n = self.order();
        let mut table = vec![vec![0usize; n]; n];
        for (i, row) in table.iter_mut().enumerate().take(n) {
            for (j, cell) in row.iter_mut().enumerate().take(n) {
                let gh = self.elements[i].compose(&self.elements[j]);
                *cell = ht[&gh.hash_key()];
            }
        }
        table
    }

    /// `group.py:260-270` — `conjugacy_table`. `conjugacy_table[g][x]` is the
    /// index of `h = x * g * x^-1`. Derived in closed form from upstream's
    /// fancy-indexed numpy expression (verified equal by hand — see
    /// `17-02-SUMMARY.md`): with `mult`/`inv` as below,
    /// `conjugacy_table[g][x] = mult[x][ mult[g][inv[x]] ]`.
    pub fn conjugacy_table(&self) -> Vec<Vec<usize>> {
        let mult = self.multiplication_table();
        let inv = self.inverse_table();
        let n = self.order();
        let mut table = vec![vec![0usize; n]; n];
        for g in 0..n {
            for x in 0..n {
                let g_xinv = mult[g][inv[x]];
                table[g][x] = mult[x][g_xinv];
            }
        }
        table
    }

    /// `group.py:276-287` — `conjugacy_mask`.
    pub fn conjugacy_mask(&self) -> Vec<Vec<bool>> {
        let table = self.conjugacy_table();
        let n = self.order();
        let mut mask = vec![vec![false; n]; n];
        for (g, row) in table.iter().enumerate() {
            for &h in row {
                mask[g][h] = true;
            }
        }
        mask
    }

    /// `group.py:289-311` — `conjugacy_classes`. Returns `(classes,
    /// representatives, inverse)` exactly as upstream: `classes[i]` is the
    /// boolean membership row for class `i`, `representatives[i]` one member
    /// element index, `inverse[e]` the class index of element `e`.
    pub fn conjugacy_classes(&self) -> (Vec<Vec<bool>>, Vec<usize>, Vec<usize>) {
        let mask = self.conjugacy_mask();
        let n = self.order();
        let mut representatives = Vec::new();
        let mut classes: Vec<Vec<bool>> = Vec::new();
        let mut inverse = vec![0usize; n];
        for g in 0..n {
            match classes.iter().position(|c| *c == mask[g]) {
                Some(ci) => inverse[g] = ci,
                None => {
                    inverse[g] = classes.len();
                    classes.push(mask[g].clone());
                    representatives.push(g);
                }
            }
        }
        (classes, representatives, inverse)
    }

    /// `group.py:313-360` — `character_table`. No lazy caching (this port
    /// recomputes on every call — group orders here are `<= 48`, and the
    /// fixed-seed random draw makes repeated calls agree up to floating
    /// rounding). `return_full` selects `chartab_full` (per-element columns)
    /// vs `chartab` (per-class columns).
    pub fn character_table(&self, return_full: bool) -> Vec<Vec<Complex64>> {
        let (classes, _representatives, inverse) = self.conjugacy_classes();
        let nclass = classes.len();
        let n = self.order();
        let class_sizes: Vec<f64> = classes
            .iter()
            .map(|c| c.iter().filter(|&&b| b).count() as f64)
            .collect();

        let mult = self.multiplication_table();
        let inv = self.inverse_table();
        let mut rng = SplitMix64::new(CHARACTER_TABLE_SEED);
        let rand: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();

        // M = classes @ rand[ginv_h] @ classes.T, then M /= class_sizes
        // (column-wise, matching numpy's trailing-axis broadcast).
        let mut m = vec![vec![0.0f64; nclass]; nclass];
        for (a, class_a) in classes.iter().enumerate() {
            for (b, class_b) in classes.iter().enumerate() {
                let mut s = 0.0;
                for (i, &in_a) in class_a.iter().enumerate() {
                    if !in_a {
                        continue;
                    }
                    for (j, &in_b) in class_b.iter().enumerate() {
                        if !in_b {
                            continue;
                        }
                        let ginv_h = mult[inv[i]][j];
                        s += rand[ginv_h];
                    }
                }
                m[a][b] = s;
            }
        }
        for row in m.iter_mut() {
            for (b, val) in row.iter_mut().enumerate() {
                *val /= class_sizes[b];
            }
        }

        let mat = faer::Mat::<f64>::from_fn(nclass, nclass, |i, j| m[i][j]);
        let eig = mat
            .eigen()
            .expect("class algebra matrix must be diagonalizable for a generic random draw");
        let u = eig.U();

        // chi[i, :] = Rchi.T[i, :] / class_sizes  ==  U.col(i) / class_sizes
        let mut chi: Vec<Vec<Complex64>> = (0..nclass)
            .map(|i| {
                let col = u.col(i);
                (0..nclass).map(|k| col[k] / class_sizes[k]).collect()
            })
            .collect();

        // normalise: unit norm w.r.t. class-size weighting, scaled by sqrt(order)
        for row in chi.iter_mut() {
            let norm: f64 = row
                .iter()
                .zip(class_sizes.iter())
                .map(|(c, &sz)| c.norm_sqr() * sz)
                .sum::<f64>()
                .sqrt();
            let scale = (n as f64).sqrt() / norm;
            for c in row.iter_mut() {
                *c *= scale;
            }
        }
        // fix the phase so column 0 (identity class) is real and positive
        for row in chi.iter_mut() {
            let c0 = row[0];
            let phase = c0 / c0.norm();
            for c in row.iter_mut() {
                *c /= phase;
            }
        }
        for row in chi.iter_mut() {
            for c in row.iter_mut() {
                *c = Complex64::new(round9(c.re), round9(c.im));
            }
        }

        // sort rows ascending, column 0 primary, negating columns 1.. first
        // (`group.py:350-353`; verified equivalent to `np.lexsort(np.rot90(.))`
        // — see 17-02-SUMMARY.md for the derivation).
        let mut order_idx: Vec<usize> = (0..nclass).collect();
        order_idx.sort_by(|&p, &q| {
            let (row_p, row_q) = (&chi[p], &chi[q]);
            for (k, (&cp0, &cq0)) in row_p.iter().zip(row_q.iter()).enumerate() {
                let sign = if k == 0 { 1.0 } else { -1.0 };
                let ord = complex_cmp(cp0 * sign, cq0 * sign);
                if ord != std::cmp::Ordering::Equal {
                    return ord;
                }
            }
            std::cmp::Ordering::Equal
        });
        let mut chi_sorted: Vec<Vec<Complex64>> =
            order_idx.iter().map(|&i| chi[i].clone()).collect();
        round_zero(&mut chi_sorted, 1e-9);

        if return_full {
            chi_sorted
                .iter()
                .map(|row| (0..n).map(|e| row[inverse[e]]).collect())
                .collect()
        } else {
            chi_sorted
        }
    }

    /// `group.py:362-371` — `project_chi`.
    pub fn project_chi(&self, chi: &[Complex64], other: &Self) -> Vec<Complex64> {
        let (i_ind, j_ind) = self.get_elements_map(other);
        let mut chi_j = vec![Complex64::new(0.0, 0.0); other.order()];
        for (i, j) in i_ind.into_iter().zip(j_ind) {
            chi_j[j] = chi[i];
        }
        chi_j
    }

    /// `group.py:373-379` — `get_elements_map`.
    pub fn get_elements_map(&self, other: &Self) -> (Vec<usize>, Vec<usize>) {
        let other_ht = other.hash_table();
        let mut i_ind = Vec::new();
        let mut j_ind = Vec::new();
        for (i, g) in self.elements.iter().enumerate() {
            if let Some(&j) = other_ht.get(&g.hash_key()) {
                i_ind.push(i);
                j_ind.push(j);
            }
        }
        (i_ind, j_ind)
    }

    /// `group.py:381-383` — `get_irrep_chi`.
    pub fn get_irrep_chi(&self, ir: usize) -> Vec<Complex64> {
        self.character_table(true)[ir].clone()
    }

    /// `group.py:173-181` — `__and__` (set intersection by hash).
    pub fn intersect(&self, other: &Self) -> Result<Self, PbcSymmError> {
        let hi: std::collections::BTreeSet<i64> =
            self.elements.iter().map(|g| g.hash_key()).collect();
        let hj: std::collections::BTreeSet<i64> =
            other.elements.iter().map(|g| g.hash_key()).collect();
        let hij: Vec<i64> = hi.intersection(&hj).copied().collect();
        Self::from_hash(&hij)
    }

    /// `group.py:183-191` — `__or__` (set union by hash).
    pub fn union(&self, other: &Self) -> Result<Self, PbcSymmError> {
        let hi: std::collections::BTreeSet<i64> =
            self.elements.iter().map(|g| g.hash_key()).collect();
        let hj: std::collections::BTreeSet<i64> =
            other.elements.iter().map(|g| g.hash_key()).collect();
        let hij: Vec<i64> = hi.union(&hj).copied().collect();
        Self::from_hash(&hij)
    }

    /// `group.py:193-200` — `issubset`.
    pub fn is_subset(&self, other: &Self) -> bool {
        let hi: std::collections::HashSet<i64> =
            self.elements.iter().map(|g| g.hash_key()).collect();
        let hj: std::collections::HashSet<i64> =
            other.elements.iter().map(|g| g.hash_key()).collect();
        hi.is_subset(&hj)
    }

    /// `group.py:385-396` — `_check_sanity`.
    fn check_sanity(&self) -> Result<(), PbcSymmError> {
        let n = self.order();
        if self.hash_table().len() != n {
            return Err(PbcSymmError::NotAGroup(
                "duplicate elements (hash collision)".into(),
            ));
        }
        let mut inv_sorted = self.inverse_table();
        inv_sorted.sort_unstable();
        if inv_sorted != (0..n).collect::<Vec<_>>() {
            return Err(PbcSymmError::NotAGroup(
                "inverse_table is not a permutation of 0..order".into(),
            ));
        }
        let mult = self.multiplication_table();
        for row in &mult {
            let mut sorted_row = row.clone();
            sorted_row.sort_unstable();
            if sorted_row != (0..n).collect::<Vec<_>>() {
                return Err(PbcSymmError::NotAGroup(
                    "multiplication_table row is not a permutation of 0..order".into(),
                ));
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// PointGroup (group.py:399-422)
// ---------------------------------------------------------------------

impl FiniteGroup<PgElement> {
    /// `group.py:403-410` — `group_name('international')`. No lazy caching
    /// (see the module doc); the Schoenflies variant is a separate method
    /// ([`Self::group_name_schoenflies`]) rather than a `notation: &str`
    /// parameter, which is a friendlier Rust shape for the same two cases
    /// upstream's `if notation.lower().startswith('scho')` distinguishes.
    pub fn group_name(&self) -> Result<&'static str, PbcSymmError> {
        let rotations: Vec<RotMatrix> = self.elements.iter().map(|e| e.matrix).collect();
        let (name, _laue) = crate::geom::get_crystal_class_from_rotations(&rotations)?;
        Ok(name)
    }

    /// `group.py:403-410` — `group_name('schoenflies')`.
    pub fn group_name_schoenflies(&self) -> Result<&'static str, PbcSymmError> {
        let name = self.group_name()?;
        tables::schoenflies(name).ok_or(PbcSymmError::UnknownCrystalClass)
    }

    /// `group.py:412-417` — `group_index`.
    pub fn group_index(&self) -> Result<usize, PbcSymmError> {
        let name = self.group_name()?;
        tables::group_index(name).ok_or(PbcSymmError::UnknownCrystalClass)
    }
}

// ---------------------------------------------------------------------
// Representation (group.py:425-476)
// ---------------------------------------------------------------------

/// `group.py:425-476` — `Representation`. Owns its `PointGroup` by value
/// (`PgElement` is `Copy`, so cloning a `<= 48`-element group is cheap)
/// instead of borrowing it, sidestepping the lifetime upstream's implicit
/// object-graph reference doesn't need to declare. `rep`/`chi` are computed
/// eagerly by the constructor used (`from_rep`/`from_chi`) rather than
/// upstream's pair of lazy properties — both directions
/// (`rep_to_chi`/`chi_to_rep`) are still ported and exercised by the
/// round-trip identity test.
#[derive(Debug, Clone)]
pub struct Representation {
    pub group: PointGroup,
    pub rep: Vec<i64>,
    pub chi: Vec<Complex64>,
}

impl Representation {
    /// `group.py:455-458` — `rep_to_chi`.
    pub fn rep_to_chi(group: &PointGroup, rep: &[i64]) -> Vec<Complex64> {
        let chartab_full = group.character_table(true);
        let n = group.order();
        let n_irrep = chartab_full.len();
        (0..n)
            .map(|i| {
                let mut s = Complex64::new(0.0, 0.0);
                for (nrow, chartab_row) in chartab_full.iter().enumerate().take(n_irrep) {
                    s += chartab_row[i] * (rep[nrow] as f64);
                }
                s
            })
            .collect()
    }

    /// `group.py:460-467` — `chi_to_rep`.
    pub fn chi_to_rep(group: &PointGroup, chi: &[Complex64]) -> Result<Vec<i64>, PbcSymmError> {
        let chartab_full = group.character_table(true);
        let order = group.order() as f64;
        let mut rep = Vec::with_capacity(chartab_full.len());
        for chartab_row in &chartab_full {
            let mut s = Complex64::new(0.0, 0.0);
            for (i, &chi_i) in chi.iter().enumerate() {
                s += chartab_row[i].conj() * chi_i;
            }
            let na = s / order;
            let rounded_re = na.re.round();
            if (na.re - rounded_re).abs() >= 1e-9 || na.im.abs() >= 1e-9 {
                return Err(PbcSymmError::NotAGroup(format!(
                    "chi_to_rep: non-integer irrep multiplicity {na:?}"
                )));
            }
            rep.push(rounded_re as i64);
        }
        Ok(rep)
    }

    pub fn from_rep(group: PointGroup, rep: Vec<i64>) -> Self {
        let chi = Self::rep_to_chi(&group, &rep);
        Self { group, rep, chi }
    }

    pub fn from_chi(group: PointGroup, chi: Vec<Complex64>) -> Result<Self, PbcSymmError> {
        let rep = Self::chi_to_rep(&group, &chi)?;
        Ok(Self { group, rep, chi })
    }

    /// `group.py:469-476` — `__matmul__`.
    pub fn matmul(&self, other: &Self) -> Result<Self, PbcSymmError> {
        let g12 = self.group.intersect(&other.group)?;
        let chi1_proj = self.group.project_chi(&self.chi, &g12);
        let chi2_proj = other.group.project_chi(&other.chi, &g12);
        let chi12: Vec<Complex64> = chi1_proj
            .iter()
            .zip(chi2_proj.iter())
            .map(|(a, b)| a * b)
            .collect();
        Self::from_chi(g12, chi12)
    }
}

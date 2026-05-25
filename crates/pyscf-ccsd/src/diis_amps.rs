//! `AmplitudeSubspace: DiisStorable` — the CCSD amplitude DIIS storable
//! (`amplitudes_to_vector`/`vector_to_amplitudes`, `ccsd.py:670`/`:679`).
//!
//! **CCSD-04 / D-06.** Packs `t1` + lower-triangular `t2` into one flat
//! error/solution vector byte-matching the upstream `amplitudes_to_vector`
//! layout, then reuses the generic `pyscf_diis::Diis<S>` machinery (NO new
//! DIIS body) with `Diis::<AmplitudeSubspace>::new(6)` (`diis_space=6`, NOT
//! SCF's 8). `DiisStorable::dot` routes through `pyscf_algebra::oracle_dot`
//! (Pitfall 9 — DIIS path drift). Packing reproduces the upstream
//! lower-triangular symmetric `t2` pack (`t2[iajb] == t2[jbia]`).
//!
//! **Upstream layout (`ccsd.py:670` `amplitudes_to_vector`):**
//! ```python
//! size        = nov + nov*(nov+1)//2          # nov = nocc*nvir
//! vector[:nov] = t1.ravel()
//! pack_tril(t2.transpose(0,2,1,3).reshape(nov,nov))   # symmetric t2[iajb]==t2[jbia]
//! ```
//! `vector_to_amplitudes` (`:679`) unpacks with `unpack_tril(filltriu=SYMMETRIC)`.
#![allow(dead_code)]

use pyscf_diis::DiisStorable;

/// Flat C-order index of `t1[i, a]` (shape `[nocc, nvir]`): `i*nvir + a`.
#[inline]
fn t1_flat(nvir: usize, i: usize, a: usize) -> usize {
    i * nvir + a
}

/// Flat C-order index of `t2[i, j, a, b]` (shape `[nocc, nocc, nvir, nvir]`):
/// `((i*nocc + j)*nvir + a)*nvir + b`.
#[inline]
fn t2_flat(nocc: usize, nvir: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * nocc + j) * nvir + a) * nvir + b
}

/// Pack `(t1, t2)` into the upstream flat DIIS vector (`ccsd.py:670`).
///
/// Layout (`nov = nocc*nvir`, total length `nov + nov*(nov+1)/2`):
/// 1. `vector[:nov] = t1.ravel()` — `t1[i,a]` at flat index `i*nvir + a`.
/// 2. The tail is `pack_tril` of the symmetric matrix
///    `M[ia, jb] = t2.transpose(0,2,1,3)[i,a,j,b] = t2[i,j,a,b]`, where the
///    row index is `ia = i*nvir + a` and the column index is `jb = j*nvir + b`.
///    `pack_tril` walks the lower triangle row-by-row: for `row` in `0..nov`,
///    for `col` in `0..=row`, emit `M[row, col]`.
///
/// `t1` is `[nocc, nvir]` C-order (`nocc*nvir` elements); `t2` is
/// `[nocc, nocc, nvir, nvir]` C-order (`nocc²·nvir²` elements).
pub fn amplitudes_to_vector(t1: &[f64], t2: &[f64], nocc: usize, nvir: usize) -> Vec<f64> {
    let nov = nocc * nvir;
    let tril = nov * (nov + 1) / 2;
    let mut v = Vec::with_capacity(nov + tril);

    // (1) t1.ravel() — already C-order [nocc, nvir].
    v.extend_from_slice(&t1[..nov]);

    // (2) pack_tril of M[ia, jb] = t2[i,j,a,b] (the transpose(0,2,1,3) view).
    //     row = ia = i*nvir + a ; col = jb = j*nvir + b ; emit when row >= col.
    for row in 0..nov {
        let i = row / nvir;
        let a = row % nvir;
        for col in 0..=row {
            let j = col / nvir;
            let b = col % nvir;
            v.push(t2[t2_flat(nocc, nvir, i, j, a, b)]);
        }
    }
    v
}

/// Unpack the flat DIIS vector back into `(t1, t2)` (`ccsd.py:679`).
///
/// Inverse of [`amplitudes_to_vector`]. The `t2` lower triangle is read from
/// the packed tail and the upper triangle is filled symmetrically
/// (`unpack_tril(filltriu=SYMMETRIC)`): `M[row, col] = M[col, row]`, i.e.
/// `t2[i,j,a,b] = t2[j,i,b,a]`. Returns `(t1, t2)` as flat C-order buffers.
pub fn vector_to_amplitudes(v: &[f64], nocc: usize, nvir: usize) -> (Vec<f64>, Vec<f64>) {
    let nov = nocc * nvir;

    // (1) t1 = vector[:nov], C-order [nocc, nvir].
    let t1 = v[..nov].to_vec();

    // (2) Fill t2 from the packed lower triangle; mirror to the upper triangle.
    let mut t2 = vec![0.0_f64; nocc * nocc * nvir * nvir];
    let mut p = nov;
    for row in 0..nov {
        let i = row / nvir;
        let a = row % nvir;
        for col in 0..=row {
            let j = col / nvir;
            let b = col % nvir;
            let val = v[p];
            p += 1;
            // M[row,col] = t2[i,j,a,b]; the symmetric partner is
            // M[col,row] = t2[j,i,b,a].
            t2[t2_flat(nocc, nvir, i, j, a, b)] = val;
            t2[t2_flat(nocc, nvir, j, i, b, a)] = val;
        }
    }
    (t1, t2)
}

/// Length of the packed amplitude vector: `nov + nov*(nov+1)/2`,
/// `nov = nocc*nvir`.
#[inline]
pub fn packed_len(nocc: usize, nvir: usize) -> usize {
    let nov = nocc * nvir;
    nov + nov * (nov + 1) / 2
}

/// CCSD amplitude subspace stored in the DIIS ring buffer (`D-06`).
///
/// Holds the flat upstream-layout vector (`t1` + lower-triangular `t2`) plus
/// the `nocc`/`nvir` shape so the kernel can `vector_to_amplitudes` after
/// extrapolation. Mirrors the SCF `FockSubspace` storable
/// (`pyscf-scf::diis_adapter`) and the test `V` storable
/// (`pyscf-diis/src/cdiis.rs:199-214`).
#[derive(Clone, Debug)]
pub struct AmplitudeSubspace {
    /// Flat packed vector (`amplitudes_to_vector` layout).
    pub packed: Vec<f64>,
    /// Number of (active) occupied orbitals.
    pub nocc: usize,
    /// Number of (active) virtual orbitals.
    pub nvir: usize,
}

impl AmplitudeSubspace {
    /// Build the subspace from `(t1, t2)` via [`amplitudes_to_vector`].
    pub fn from_amplitudes(t1: &[f64], t2: &[f64], nocc: usize, nvir: usize) -> Self {
        Self {
            packed: amplitudes_to_vector(t1, t2, nocc, nvir),
            nocc,
            nvir,
        }
    }

    /// Unpack the stored vector back into `(t1, t2)` via
    /// [`vector_to_amplitudes`].
    pub fn to_amplitudes(&self) -> (Vec<f64>, Vec<f64>) {
        vector_to_amplitudes(&self.packed, self.nocc, self.nvir)
    }
}

impl DiisStorable for AmplitudeSubspace {
    fn as_flat(&self) -> &[f64] {
        &self.packed
    }
    fn from_flat(&mut self, s: &[f64]) {
        self.packed.copy_from_slice(s);
    }
    fn dot(&self, other: &Self) -> f64 {
        // MANDATORY: route through oracle_dot for bit-identical cross-platform
        // reductions (Pitfall 9 — DIIS path drift). NOT iter().sum().
        pyscf_algebra::oracle_dot(&self.packed, &other.packed)
    }
    fn len(&self) -> usize {
        self.packed.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `packed_len == nov + nov*(nov+1)/2` and `as_flat().len()` matches it.
    #[test]
    fn len_is_nov_plus_tril() {
        let (nocc, nvir) = (2usize, 3usize);
        let nov = nocc * nvir; // 6
        let expected = nov + nov * (nov + 1) / 2; // 6 + 21 = 27
        assert_eq!(packed_len(nocc, nvir), expected);

        let t1 = vec![0.0_f64; nov];
        let t2 = vec![0.0_f64; nocc * nocc * nvir * nvir];
        let sub = AmplitudeSubspace::from_amplitudes(&t1, &t2, nocc, nvir);
        assert_eq!(sub.len(), expected);
        assert_eq!(sub.as_flat().len(), expected);
    }

    /// Byte-match against a hand-built reference vector for nocc=1, nvir=2.
    ///
    /// nov = 2. t1 = [t1_00, t1_01]. M[ia,jb] = t2[0,0,a,b] (only i=j=0).
    /// M is 2x2 symmetric:
    ///   M[0,0] = t2[0,0,0,0],  M[1,0] = t2[0,0,1,0],  M[1,1] = t2[0,0,1,1]
    /// pack_tril row-by-row (row, col<=row): [M00, M10, M11].
    /// Expected vector = [t1_00, t1_01, M00, M10, M11].
    #[test]
    fn pack_byte_matches_hand_reference() {
        let (nocc, nvir) = (1usize, 2usize);
        // t1[i,a] flat = i*nvir + a.
        let t1 = vec![10.0, 11.0]; // t1[0,0]=10, t1[0,1]=11
        // t2[i,j,a,b] flat = ((i*nocc+j)*nvir+a)*nvir+b. nocc=1 so i=j=0.
        // indices: (0,0,0,0)=0, (0,0,0,1)=1, (0,0,1,0)=2, (0,0,1,1)=3.
        // Make t2 SYMMETRIC under [iajb]==[jbia]: with i=j=0 that means
        // t2[0,0,a,b] == t2[0,0,b,a], so t2[0,0,0,1] must equal t2[0,0,1,0].
        let mut t2 = vec![0.0_f64; nocc * nocc * nvir * nvir];
        t2[0] = 20.0; // M[0,0]
        t2[1] = 21.0; // M[0,1] (== M[1,0] by symmetry)
        t2[2] = 21.0; // M[1,0]
        t2[3] = 22.0; // M[1,1]

        let v = amplitudes_to_vector(&t1, &t2, nocc, nvir);
        // [t1_00, t1_01, M00, M10, M11] = [10, 11, 20, 21, 22].
        assert_eq!(v, vec![10.0, 11.0, 20.0, 21.0, 22.0]);
        assert_eq!(v.len(), packed_len(nocc, nvir)); // 2 + 3 = 5
    }

    /// Round-trip: pack then unpack reproduces (t1, t2) bit-identically for a
    /// SYMMETRIC t2 (the converged-amplitude invariant t2[iajb]==t2[jbia]).
    #[test]
    fn round_trip_bit_identical() {
        let (nocc, nvir) = (2usize, 3usize);
        let nov = nocc * nvir;
        // Arbitrary t1.
        let mut t1 = vec![0.0_f64; nov];
        for (k, x) in t1.iter_mut().enumerate() {
            *x = 1.0 + k as f64 * 0.5;
        }
        // Build a SYMMETRIC t2: t2[i,j,a,b] = t2[j,i,b,a]. Symmetrize from a
        // seed so the round-trip is exact.
        let mut t2 = vec![0.0_f64; nocc * nocc * nvir * nvir];
        let mut seed = vec![0.0_f64; t2.len()];
        for (k, x) in seed.iter_mut().enumerate() {
            *x = (k as f64).sin();
        }
        for i in 0..nocc {
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let f = t2_flat(nocc, nvir, i, j, a, b);
                        let g = t2_flat(nocc, nvir, j, i, b, a);
                        t2[f] = 0.5 * (seed[f] + seed[g]);
                    }
                }
            }
        }

        let v = amplitudes_to_vector(&t1, &t2, nocc, nvir);
        let (t1b, t2b) = vector_to_amplitudes(&v, nocc, nvir);
        assert_eq!(t1, t1b, "t1 must round-trip bit-identically");
        assert_eq!(t2, t2b, "symmetric t2 must round-trip bit-identically");
    }

    /// `AmplitudeSubspace::dot` equals `oracle_dot` of the two packed vectors
    /// (NOT iter().sum()) — Pitfall 9.
    #[test]
    fn dot_equals_oracle_dot() {
        let (nocc, nvir) = (2usize, 2usize);
        let nov = nocc * nvir;
        let t1a = vec![0.3_f64; nov];
        let t1b = vec![-0.2_f64; nov];
        let t2a = vec![0.1_f64; nocc * nocc * nvir * nvir];
        let t2b = vec![0.05_f64; nocc * nocc * nvir * nvir];
        let sa = AmplitudeSubspace::from_amplitudes(&t1a, &t2a, nocc, nvir);
        let sb = AmplitudeSubspace::from_amplitudes(&t1b, &t2b, nocc, nvir);

        let got = sa.dot(&sb);
        let expected = pyscf_algebra::oracle_dot(&sa.packed, &sb.packed);
        assert_eq!(
            got.to_bits(),
            expected.to_bits(),
            "dot must be bit-identical to oracle_dot (Pitfall 9)"
        );
    }
}

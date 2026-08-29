//! The McMurchie–Davidson Hermite expansion — HOST side (plan 13-01).
//!
//! The product of two primitive Cartesian Gaussians expands in Hermite
//! Gaussians centred at the product centre `P`:
//!
//! ```text
//! φ_i(x−A) · φ_j(x−B) = Σ_{t=0}^{i+j} E_t^{ij} · Λ_t(x; P, p)
//! ```
//!
//! with (`x_PA = P − A`, `x_PB = P − B`, `p = a + b`)
//!
//! ```text
//! E_0^{00}    = K_AB = exp(−(a·b/p)·|A−B|²)
//! E_t^{i+1,j} = (1/(2p))·E_{t−1}^{ij} + x_PA·E_t^{ij} + (t+1)·E_{t+1}^{ij}
//! E_t^{i,j+1} = (1/(2p))·E_{t−1}^{ij} + x_PB·E_t^{ij} + (t+1)·E_{t+1}^{ij}
//! E_t^{ij}    = 0   for t < 0 or t > i+j
//! ```
//!
//! Because `∫ dx e^{−iG·x} Λ_t(x;P,p) = (−iG)^t · √(π/p) · e^{−G²/4p} · e^{−iG·P}`
//! per axis, the whole Fourier transform of an AO pair is a polynomial in `G`
//! whose coefficients are these `E`s. That is the entire reason the recursion
//! runs here and not on the device: `E` is independent of `G`, and there are
//! `O(nG)` more G-vectors than primitive pairs.
//!
//! **There is exactly one copy of this recursion in the workspace.** Plan 13-02's
//! single-centre `ft_ao` calls [`e_coefficients`] with `lj = 0` rather than
//! writing a second one.

/// Flattened `E[i][j][t]` for one Cartesian axis.
///
/// Layout is row-major `i · (lj+1) · (lij+1) + j · (lij+1) + t` with
/// `lij = li + lj`; [`e_index`] is the accessor and the device kernel indexes
/// the uploaded buffer the same way.
#[derive(Debug, Clone, PartialEq)]
pub struct ETable {
    /// Bra angular momentum on this axis.
    pub li: u32,
    /// Ket angular momentum on this axis.
    pub lj: u32,
    /// `(li+1)·(lj+1)·(li+lj+1)` coefficients.
    pub data: Vec<f64>,
}

impl ETable {
    /// Stride of the `t` axis — `li + lj + 1`.
    #[inline]
    pub fn nt(&self) -> usize {
        (self.li + self.lj + 1) as usize
    }
    /// `E[i][j][t]`.
    #[inline]
    pub fn get(&self, i: u32, j: u32, t: u32) -> f64 {
        self.data[e_index(self.lj, self.li + self.lj, i, j, t)]
    }
}

/// Flat index of `E[i][j][t]` in an [`ETable`] built for `(li, lj)`.
///
/// Taken as a free function so the device kernel and the host agree on one
/// formula; `lj` and `lij = li + lj` are the only shape parameters it needs.
#[inline]
pub fn e_index(lj: u32, lij: u32, i: u32, j: u32, t: u32) -> usize {
    let njt = ((lj + 1) * (lij + 1)) as usize;
    let nt = (lij + 1) as usize;
    i as usize * njt + j as usize * nt + t as usize
}

/// Number of coefficients an [`ETable`] for `(li, lj)` holds.
#[inline]
pub fn e_len(li: u32, lj: u32) -> usize {
    ((li + 1) * (lj + 1) * (li + lj + 1)) as usize
}

/// Build the one-axis Hermite expansion coefficients.
///
/// * `li`, `lj` — the maximum bra/ket power on this axis.
/// * `p` — the combined exponent `a + b`.
/// * `xpa`, `xpb` — `P − A` and `P − B` on this axis.
/// * `seed` — `E_0^{00}`. Pass `K_AB` on ONE axis and `1.0` on the other two so
///   the pre-exponential factor is counted exactly once (a doubled `K_AB` is
///   the classic way to get this wrong and it survives every s-only test).
pub fn e_coefficients(li: u32, lj: u32, p: f64, xpa: f64, xpb: f64, seed: f64) -> ETable {
    let lij = li + lj;
    let mut data = vec![0.0f64; e_len(li, lj)];
    let half_inv_p = 0.5 / p;
    data[e_index(lj, lij, 0, 0, 0)] = seed;

    // Walk i first at j = 0, then raise j for every i. Every read is from a
    // strictly smaller (i+j), so one pass in this order suffices.
    for i in 0..=li {
        for j in 0..=lj {
            if i == 0 && j == 0 {
                continue;
            }
            let (pi, pj, x) = if j > 0 {
                (i, j - 1, xpb) // raise j from (i, j-1)
            } else {
                (i - 1, j, xpa) // raise i from (i-1, 0)
            };
            let tmax = i + j;
            for t in 0..=tmax {
                let mut v = x * data[e_index(lj, lij, pi, pj, t)];
                if t > 0 {
                    v += half_inv_p * data[e_index(lj, lij, pi, pj, t - 1)];
                }
                if t < pi + pj {
                    v += (t + 1) as f64 * data[e_index(lj, lij, pi, pj, t + 1)];
                }
                data[e_index(lj, lij, i, j, t)] = v;
            }
        }
    }
    ETable { li, lj, data }
}

/// `K_AB = exp(−(a·b/(a+b))·|A−B|²)`, the pre-exponential factor.
#[inline]
pub fn k_ab(a: f64, b: f64, ab2: f64) -> f64 {
    (-(a * b / (a + b)) * ab2).exp()
}

//! Density matrices and electronic energies — `khf.py:238-268`,
//! `kuhf.py:46-60`, `kuhf.py:205-226`.

use pyscf_algebra::{CTensor, oracle_sum};

use crate::types::{KDms, KMats};

/// `make_rdm1(mo_coeff_kpts, mo_occ_kpts)` — `khf.py:238-247`.
///
/// `D^k = sum_i occ_i C^k[:, i] C^k[:, i]^H`, one matrix per k-point,
/// ROW-MAJOR out. `mo_coeff` is COLUMN-MAJOR `nao x nmo`.
///
/// The construction is explicitly Hermitian: entry `(mu, nu)` and `(nu, mu)`
/// are computed as conjugates of one product rather than independently, so a
/// converged `D` is exactly Hermitian and the DIIS error vector `FDS - SDF`
/// vanishes cleanly.
pub fn make_rdm1(mo_coeff: &[CTensor], mo_occ: &[Vec<f64>], nao: usize) -> KMats {
    mo_coeff
        .iter()
        .zip(mo_occ.iter())
        .map(|(c, occ)| make_rdm1_one(c, occ, nao))
        .collect()
}

/// One k-point's density matrix.
pub fn make_rdm1_one(c: &CTensor, occ: &[f64], nao: usize) -> CTensor {
    let mut re = vec![0.0_f64; nao * nao];
    let mut im = vec![0.0_f64; nao * nao];
    for (i, o) in occ.iter().enumerate() {
        if *o == 0.0 {
            continue;
        }
        let base = i * nao;
        for mu in 0..nao {
            let (ar, ai) = (c.re[base + mu], c.im[base + mu]);
            for nu in 0..nao {
                // C[mu, i] * conj(C[nu, i])
                let (br, bi) = (c.re[base + nu], -c.im[base + nu]);
                re[mu * nao + nu] += o * (ar * br - ai * bi);
                im[mu * nao + nu] += o * (ar * bi + ai * br);
            }
        }
    }
    CTensor::from_planes(re, im)
}

/// `Tr(A B)` over a row-major `n x n` pair — `einsum('ij,ji->', a, b)`.
///
/// # D-PBC-17 — the reduction is ORDERED
///
/// This is the SECOND copy of the routine (`pyscf-pbc-dft::veff::trace_ab` is
/// the other; they are on opposite sides of the ALG-06 crate split, which is
/// the only reason there are two). The D-PBC-17 pass that ordered the `veff.rs`
/// copy did not reach this one, and this is the copy [`energy_elec`] uses — so
/// until U-03 the ONE-ELECTRON energy `e1`, and every `Kuks`/`Krks`/`Kroks`
/// total built on it, still came off a naive `n^2`-long running sum.
///
/// The `n^2` products are materialised in a FIXED index order (`i`-major,
/// `j`-minor — the order the pre-existing loop accumulated in) and each plane
/// goes through [`oracle_sum`], so the recursion-tree shape depends only on
/// `n^2` and the fixed `PAIRWISE_CHUNK`. For `n^2 <= PAIRWISE_CHUNK`
/// (`nao <= 11`, which covers the `nao = 8` reference cells) `oracle_sum`'s
/// base case is a strict left-to-right fold from `0.0` — **bit-identical** to
/// the loop this replaced, so the existing gates do not move. For `nao >= 12`
/// the tree engages and the bound improves from `O(n^2 eps)` to
/// `O(log2(n^2) eps)`.
pub fn trace_ab(a: &CTensor, b: &CTensor, n: usize) -> (f64, f64) {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n * n);
    let mut tr = vec![0.0_f64; n * n];
    let mut ti = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = (a.re[i * n + j], a.im[i * n + j]);
            let (br, bi) = (b.re[j * n + i], b.im[j * n + i]);
            tr[i * n + j] = ar * br - ai * bi;
            ti[i * n + j] = ar * bi + ai * br;
        }
    }
    (oracle_sum(&tr), oracle_sum(&ti))
}

/// `energy_elec(mf, dm_kpts, h1e_kpts, vhf_kpts)` — `khf.py:249-268` (RHF) and
/// `kuhf.py:205-226` (UHF; the two channels are simply summed).
///
/// ```text
/// e1     = (1/nkpts) sum_{s,k} Tr(D^{s,k} H^k)
/// e_coul = (1/nkpts) sum_{s,k} Tr(D^{s,k} V^{s,k}) / 2
/// ```
///
/// `h1e` is shared by both channels, exactly as upstream's `h1e_kpts` is.
/// Returns `(e_elec, e_coul)`, both REAL — the imaginary parts cancel by
/// Hermiticity, and a large residue means the Coulomb integrals have not
/// converged (upstream warns; the caller can inspect [`coulomb_imag`]).
pub fn energy_elec(dms: &KDms, h1e: &KMats, vhf: &KDms, nao: usize) -> (f64, f64) {
    let nkpts = h1e.len();
    let inv = 1.0 / nkpts as f64;
    // The `1/nkpts` is applied ONCE to the finished k-sum, as upstream's
    // `1./nkpts * np.einsum('kij,kji', ...)` does, not inside the loop: the two
    // differ in the last bits and the k-sum is the quantity upstream rounds.
    //
    // D-PBC-17 (U-03 step 2): the `nset * nkpts`-long outer chain is collected
    // and reduced with the ordered tree, not folded with a running `+=`. Two
    // ordered reductions compose into an ordered reduction, whereas an ordered
    // inner sum folded by a naive outer loop is only as good as the outer loop.
    let n = dms.len() * nkpts;
    let mut e1_parts = Vec::with_capacity(n);
    let mut ecoul_parts = Vec::with_capacity(n);
    for (set, dmset) in dms.iter().enumerate() {
        for k in 0..nkpts {
            e1_parts.push(trace_ab(&dmset[k], &h1e[k], nao).0);
            ecoul_parts.push(trace_ab(&dmset[k], &vhf[set][k], nao).0);
        }
    }
    let e1 = inv * oracle_sum(&e1_parts);
    let e_coul = inv * oracle_sum(&ecoul_parts) * 0.5;
    (e1 + e_coul, e_coul)
}

/// The imaginary residue of `e_coul` — upstream's `CHECK_COULOMB_IMAG` warning
/// (`khf.py:263-267`), exposed so a caller can assert on it instead of reading
/// a log line.
pub fn coulomb_imag(dms: &KDms, vhf: &KDms, nao: usize) -> f64 {
    let nkpts = vhf[0].len();
    let inv = 1.0 / nkpts as f64;
    let mut parts = Vec::with_capacity(dms.len() * nkpts);
    for (set, dmset) in dms.iter().enumerate() {
        for k in 0..nkpts {
            parts.push(trace_ab(&dmset[k], &vhf[set][k], nao).1);
        }
    }
    inv * oracle_sum(&parts) * 0.5
}

/// Number of electrons implied by a density matrix: `sum_k Tr(D^k S^k) / nkpts`
/// summed over channels — the check `khf.py:838-852` runs on the initial guess.
pub fn electron_count(dms: &KDms, s1e: &KMats, nao: usize) -> f64 {
    oracle_sum(&electron_count_per_set(dms, s1e, nao))
}

/// The same count, resolved PER CHANNEL — `lib.einsum('xkij,kji->x', dm, s1e)`
/// (`kuhf.py:476`), whose length is `nset`.
///
/// U-02: the unrestricted initial guess is renormalised against `(nalpha,
/// nbeta)` with a SEPARATE factor per channel. Summing the two channels into
/// one `f64` first, as [`electron_count`] does, throws that information away —
/// and on a `cell.spin != 0` cell the per-channel factors are the ONLY thing
/// that polarises the minao guess at all.
pub fn electron_count_per_set(dms: &KDms, s1e: &KMats, nao: usize) -> Vec<f64> {
    dms.iter()
        .map(|dmset| {
            let parts: Vec<f64> = s1e
                .iter()
                .enumerate()
                .map(|(k, s)| trace_ab(&dmset[k], s, nao).0)
                .collect();
            oracle_sum(&parts)
        })
        .collect()
}

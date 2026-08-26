//! Density matrices and electronic energies — `khf.py:238-268`,
//! `kuhf.py:46-60`, `kuhf.py:205-226`.

use pyscf_algebra::CTensor;

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
pub fn trace_ab(a: &CTensor, b: &CTensor, n: usize) -> (f64, f64) {
    let mut sr = 0.0_f64;
    let mut si = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let (ar, ai) = (a.re[i * n + j], a.im[i * n + j]);
            let (br, bi) = (b.re[j * n + i], b.im[j * n + i]);
            sr += ar * br - ai * bi;
            si += ar * bi + ai * br;
        }
    }
    (sr, si)
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
    let mut e1 = 0.0_f64;
    let mut e_coul = 0.0_f64;
    for (set, dmset) in dms.iter().enumerate() {
        for k in 0..nkpts {
            e1 += trace_ab(&dmset[k], &h1e[k], nao).0;
            e_coul += trace_ab(&dmset[k], &vhf[set][k], nao).0;
        }
    }
    let e1 = inv * e1;
    let e_coul = inv * e_coul * 0.5;
    (e1 + e_coul, e_coul)
}

/// The imaginary residue of `e_coul` — upstream's `CHECK_COULOMB_IMAG` warning
/// (`khf.py:263-267`), exposed so a caller can assert on it instead of reading
/// a log line.
pub fn coulomb_imag(dms: &KDms, vhf: &KDms, nao: usize) -> f64 {
    let nkpts = vhf[0].len();
    let inv = 1.0 / nkpts as f64;
    let mut im = 0.0_f64;
    for (set, dmset) in dms.iter().enumerate() {
        for k in 0..nkpts {
            im += trace_ab(&dmset[k], &vhf[set][k], nao).1;
        }
    }
    inv * im * 0.5
}

/// Number of electrons implied by a density matrix: `sum_k Tr(D^k S^k) / nkpts`
/// summed over channels — the check `khf.py:838-852` runs on the initial guess.
pub fn electron_count(dms: &KDms, s1e: &KMats, nao: usize) -> f64 {
    let mut ne = 0.0_f64;
    for dmset in dms {
        for (k, s) in s1e.iter().enumerate() {
            ne += trace_ab(&dmset[k], s, nao).0;
        }
    }
    ne
}

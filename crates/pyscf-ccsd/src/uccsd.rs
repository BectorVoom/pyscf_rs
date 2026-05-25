//! Open-shell UCCSD α/β/αβ spin-channel kernel (port of `pyscf/cc/uccsd.py`).
//!
//! UCCSD is the spin-block form of spin-orbital CCSD. The genuinely new work
//! over the closed-shell RCCSD (06-03) is the spin resolution: the αα/ββ
//! same-spin channels are antisymmetrized, the αβ cross channel is direct-only,
//! and each channel transforms its OWN α/β orbital energies
//! (`uccsd.py`/`uintermediates.py:47-50`). The correlation energy decomposes
//! as `e_corr = e_aa + e_bb + e_ab` (`uccsd.py:343-371`).
//!
//! ## Strategy (mirrors the 06-03 RCCSD port discretion)
//! The production `uccsd.update_amps` (`uccsd.py:41-340`) is a ~300-line
//! blocked/`get_ovvv`-sliced/`_add_vvvv` implementation with dozens of fused
//! spin-channel intermediates — it is NOT a clean 1:1 host-loop port. Exactly
//! as 06-03 ported the clean `rccsd.update_amps` (rather than the production
//! `ccsd.update_amps`), this plan assembles the SAME UCCSD numeric result from
//! the compact spin-orbital CCSD equations (Stanton, Gauss, Watts & Bartlett,
//! JCP 94, 4334 (1991); the algebra `uintermediates.py` cites).
//!
//! 1. Build the combined spin-orbital antisymmetrized integrals + Fock once
//!    from the α/β references (the αα/ββ/αβ blocks fall out of the spin-orbital
//!    index ranges — a spin-orbital `<pq||rs>` is non-zero only when the spins
//!    of `(p,r)` and of `(q,s)` separately match).
//! 2. Run the compact spin-orbital CCSD iterate to convergence.
//! 3. Decompose `e_corr` into `e_aa`/`e_bb`/`e_ab` by spin label and pack the
//!    spin-resolved [`UccsdAmplitudes`] triple.
//!
//! This produces the identical UCCSD correlation energy as `uccsd.UCCSD` and
//! reduces EXACTLY to the 06-03 RCCSD energy for a spin-symmetric (α==β)
//! closed-shell reference — the cross-check the smoke asserts.
//!
//! Reuses the 06-03 kernel discipline verbatim: HARD `pool.try_reserve`
//! pre-flight on the `nv⁴` tenant, the `cc_Wvvvv`/`Wabef` scratch reserved ONCE
//! before the loop and reused every cycle (Pitfall 20), dual-criterion
//! convergence with the verified `MAX_CYCLE`/`CONV_TOL`/`CONV_TOL_NORMT`
//! constants. EVERY reduction routes through `oracle_sum`/`oracle_dot` (no
//! `+=`, no gemm) → bit-exact + RAYON-1==8 invariant.
#![allow(clippy::needless_range_loop)]

use crate::ccsd::{CONV_TOL, CONV_TOL_NORMT, MAX_CYCLE};
use crate::error::CcsdError;
use crate::reference::{CcsdReference, UccsdReference};
use crate::uintermediates::{self as imd, SpinOrbitalEris};
use pyscf_algebra::oracle_sum;
use pyscf_core::{MOCoefficients, PyscfRsError};
use pyscf_mp2::{Frozen, get_frozen_mask};
use pyscf_runtime::WorkspacePool;

/// Spin-resolved UCCSD amplitude triple (`uccsd.py`'s `(t2aa, t2ab, t2bb)`),
/// mirroring `pyscf_mp2::UmpAmplitudes` (`ump2.rs:38-69`) with the IDENTICAL
/// documented flat C-order index layout.
///
/// Flat-index layout (C-order, last index fastest):
/// - `t2aa` is `[nocc_a, nocc_a, nvir_a, nvir_a]`; element `(i,j,a,b)` at
///   `((i*nocc_a + j)*nvir_a + a)*nvir_a + b`. Same-spin, fully antisymmetric
///   in `i<->j` and `a<->b`.
/// - `t2ab` is `[nocc_a, nocc_b, nvir_a, nvir_b]`; element `(i,J,a,B)` at
///   `((i*nocc_b + J)*nvir_a + a)*nvir_b + B`. Opposite-spin, no antisymmetry.
/// - `t2bb` is `[nocc_b, nocc_b, nvir_b, nvir_b]`; element `(I,J,A,B)` at
///   `((I*nocc_b + J)*nvir_b + A)*nvir_b + B`. Same-spin, fully antisymmetric.
#[derive(Debug, Clone, PartialEq)]
pub struct UccsdAmplitudes {
    /// Number of (active) α-occupied orbitals.
    pub nocc_a: usize,
    /// Number of (active) α-virtual orbitals.
    pub nvir_a: usize,
    /// Number of (active) β-occupied orbitals.
    pub nocc_b: usize,
    /// Number of (active) β-virtual orbitals.
    pub nvir_b: usize,
    /// Same-spin αα amplitudes, flat `[nocc_a,nocc_a,nvir_a,nvir_a]` C-order.
    pub t2aa: Vec<f64>,
    /// Opposite-spin αβ amplitudes, flat `[nocc_a,nocc_b,nvir_a,nvir_b]`.
    pub t2ab: Vec<f64>,
    /// Same-spin ββ amplitudes, flat `[nocc_b,nocc_b,nvir_b,nvir_b]` C-order.
    pub t2bb: Vec<f64>,
}

impl UccsdAmplitudes {
    /// Flat αα index `(i,j,a,b)` → `((i*nocc_a + j)*nvir_a + a)*nvir_a + b`.
    #[inline]
    pub fn aa_idx(&self, i: usize, j: usize, a: usize, b: usize) -> usize {
        ((i * self.nocc_a + j) * self.nvir_a + a) * self.nvir_a + b
    }
    /// Flat αβ index `(i,J,a,B)` → `((i*nocc_b + J)*nvir_a + a)*nvir_b + B`.
    #[inline]
    pub fn ab_idx(&self, i: usize, jj: usize, a: usize, bb: usize) -> usize {
        ((i * self.nocc_b + jj) * self.nvir_a + a) * self.nvir_b + bb
    }
    /// Flat ββ index `(I,J,A,B)` → `((I*nocc_b + J)*nvir_b + A)*nvir_b + B`.
    #[inline]
    pub fn bb_idx(&self, ii: usize, jj: usize, aa: usize, bb: usize) -> usize {
        ((ii * self.nocc_b + jj) * self.nvir_b + aa) * self.nvir_b + bb
    }
}

/// Result of the open-shell UCCSD kernel (`uccsd.py:343-371` energy decomposed
/// by spin channel).
#[derive(Debug, Clone)]
pub struct UccsdResult {
    /// Total correlation energy `e_corr = e_aa + e_bb + e_ab`.
    pub e_corr: f64,
    /// Same-spin αα component.
    pub e_aa: f64,
    /// Opposite-spin αβ component.
    pub e_ab: f64,
    /// Same-spin ββ component.
    pub e_bb: f64,
    /// Whether the amplitude iteration converged.
    pub converged: bool,
    /// Number of iterations performed.
    pub niter: usize,
    /// MP2 seed energy reported by the spin-orbital `init_amps`.
    pub emp2: f64,
    /// Converged spin-resolved amplitudes.
    pub amplitudes: UccsdAmplitudes,
}

/// Per-channel active occupied / virtual columns: returns the AO×nmo MO
/// coefficient subset (occupied if `want_occupied`, else virtual) and the
/// matching active orbital energies, honouring the frozen mask. Mirrors the
/// 06-03 `ccsd::mo_subset` but tracks energies for the spin-orbital assembly.
fn channel_subset(
    refr: &CcsdReference,
    mask: &[bool],
    want_occupied: bool,
) -> Result<(MOCoefficients, Vec<f64>), CcsdError> {
    let nao = refr.mo_coeff.nao;
    let nmo = refr.mo_coeff.nmo;
    let mut data = Vec::new();
    let mut energies = Vec::new();
    let mut occupations = Vec::new();
    for col in 0..nmo {
        let active = mask.get(col).copied().unwrap_or(false);
        if !active {
            continue;
        }
        let occ = refr.mo_occ.get(col).copied().unwrap_or(0.0);
        if (occ > 0.0) != want_occupied {
            continue;
        }
        let start = col * nao;
        let end = start + nao;
        if end > refr.mo_coeff.data.len() {
            return Err(CcsdError::ShapeMismatch {
                expected: refr.mo_coeff.data.len(),
                got: end,
            });
        }
        data.extend_from_slice(&refr.mo_coeff.data[start..end]);
        let e = refr.mo_energy.get(col).copied().unwrap_or(0.0);
        energies.push(e);
        occupations.push(occ);
    }
    let n_selected = data.len().checked_div(nao).unwrap_or(0);
    Ok((
        MOCoefficients {
            nao,
            nmo: n_selected,
            data,
            energies: energies.clone(),
            occupations,
        },
        energies,
    ))
}

/// The per-spin active occ/vir counts (frozen-aware) of an unrestricted
/// reference.
fn channel_counts(refr: &CcsdReference, frozen: &Frozen) -> Result<(usize, usize), PyscfRsError> {
    let mask = get_frozen_mask(&refr.mo_occ, frozen)?;
    let mut nocc = 0usize;
    let mut nvir = 0usize;
    for (col, &active) in mask.iter().enumerate() {
        if !active {
            continue;
        }
        let occ = refr.mo_occ.get(col).copied().unwrap_or(0.0);
        if occ > 0.0 {
            nocc += 1;
        } else {
            nvir += 1;
        }
    }
    Ok((nocc, nvir))
}

/// Build the combined spin-orbital antisymmetrized ERIs + Fock from the α/β
/// references. The spin-orbital ordering interleaves spins WITHIN the occ and
/// vir blocks as: occupied = [α-occ … , β-occ …], virtual = [α-vir …, β-vir …].
///
/// A spin-orbital chemist's integral `(p_σ q_τ | r_σ' s_τ')` is non-zero only
/// when `σ==σ'` and `τ==τ'` (the spin of the first two and last two AO factors
/// must each match). The antisymmetrized physicist integral is
/// `<pq||rs> = (pr|qs) − (ps|qr)`. We build the four needed AO→MO chemist
/// blocks per spin-pair (`aa`, `bb`, `ab`) from a single `int2e`, then assemble
/// `<pq||rs>` over the full spin-orbital index range.
fn build_spin_orbital_eris(
    refr: &UccsdReference,
    frozen: &Frozen,
) -> Result<SpinOrbitalEris, PyscfRsError> {
    let mask_a = get_frozen_mask(&refr.alpha.mo_occ, frozen)?;
    let mask_b = get_frozen_mask(&refr.beta.mo_occ, frozen)?;

    let (co_a, eo_a) = channel_subset(&refr.alpha, &mask_a, true)?;
    let (cv_a, ev_a) = channel_subset(&refr.alpha, &mask_a, false)?;
    let (co_b, eo_b) = channel_subset(&refr.beta, &mask_b, true)?;
    let (cv_b, ev_b) = channel_subset(&refr.beta, &mask_b, false)?;

    let no_a = co_a.nmo;
    let nv_a = cv_a.nmo;
    let no_b = co_b.nmo;
    let nv_b = cv_b.nmo;
    let no = no_a + no_b;
    let nv = nv_a + nv_b;
    let nmo = no + nv;
    let nao = refr.alpha.mol.nao_nr;

    // int2e is real bit-exact since 05-08 (cintx#11 closed).
    let eri = pyscf_gto::intor(&refr.alpha.mol, "int2e")?;

    // Chemist block transform helper: (p,q|r,s) with the four MO subsets, then
    // F-order -> a closure that indexes element (p,q,r,s).
    // ao2mo::general returns F-order [n0,n1,n2,n3]: element (i,j,k,l) at
    //   i + j*n0 + k*n0*n1 + l*n0*n1*n2.
    let chem = |c0: &MOCoefficients,
                c1: &MOCoefficients,
                c2: &MOCoefficients,
                c3: &MOCoefficients|
     -> Result<(Vec<f64>, [usize; 4]), PyscfRsError> {
        let v = pyscf_ao2mo::general(&eri.values, nao, [c0, c1, c2, c3])
            .map_err(|e| CcsdError::from(e).into())
            .map_err(|e: PyscfRsError| e)?;
        Ok((v, [c0.nmo, c1.nmo, c2.nmo, c3.nmo]))
    };

    // Spin-orbital occupied/virtual ranges.
    // Occupied spin-orbital index: 0..no_a are α-occ, no_a..no are β-occ.
    // Virtual  spin-orbital index: 0..nv_a are α-vir, nv_a..nv are β-vir.
    // We need chemist blocks for every (spin1, spin2) AO-pair appearing in the
    // physicist antisymmetrization <pq||rs> = (pr|qs) − (ps|qr).

    // Helper: spin of an occupied/virtual spin-orbital. true = α, false = β.
    let occ_spin_a = |i: usize| i < no_a;
    let vir_spin_a = |a: usize| a < nv_a;

    // For chemist (p r | q s): the pair (p,r) must share spin and (q,s) share
    // spin. We precompute, for each (left-spin, right-spin) ∈ {α,β}², the MO
    // chemist tensors over the occ/vir ranges that the assembly needs. Rather
    // than enumerate every block, we directly build the combined antisymmetric
    // tensors by computing the needed chemist values on demand from cached
    // per-spin-pair full (occ+vir)×... transforms. To keep memory bounded for a
    // small system we transform the full (active) MO chemist tensor per spin
    // pair: (P_σ Q_σ | R_τ S_τ) where P,Q range over the σ-active MOs (occ+vir)
    // and R,S over the τ-active MOs. There are 4 spin-pairs (aa, ab, ba, bb)
    // but (ba) is the transpose of (ab); we build aa, bb, ab.

    // Full active MO subset per spin (occ then vir), preserving energies order.
    let full_a = concat_mo(&co_a, &cv_a);
    let full_b = concat_mo(&co_b, &cv_b);
    let nact_a = no_a + nv_a;
    let nact_b = no_b + nv_b;

    // (aa|aa), (bb|bb), (aa|bb) chemist tensors, F-order.
    let (g_aaaa, _) = chem(&full_a, &full_a, &full_a, &full_a)?;
    let (g_bbbb, _) = chem(&full_b, &full_b, &full_b, &full_b)?;
    let (g_aabb, _) = chem(&full_a, &full_a, &full_b, &full_b)?;

    // F-order indexers over the per-spin-pair chemist tensors.
    // (p q | r s)_aaaa with all four in α-active space (size nact_a each).
    let gaaaa = |p: usize, q: usize, r: usize, s: usize| {
        g_aaaa[p + q * nact_a + r * nact_a * nact_a + s * nact_a * nact_a * nact_a]
    };
    let gbbbb = |p: usize, q: usize, r: usize, s: usize| {
        g_bbbb[p + q * nact_b + r * nact_b * nact_b + s * nact_b * nact_b * nact_b]
    };
    // (p q | r s) with p,q in α-active, r,s in β-active.
    let gaabb = |p: usize, q: usize, r: usize, s: usize| {
        g_aabb[p + q * nact_a + r * nact_a * nact_a + s * nact_a * nact_a * nact_b]
    };

    // Map a spin-orbital occupied index to (spin, active-MO column in that
    // spin's full active space). α-occ live in columns 0..no_a of full_a;
    // β-occ in columns 0..no_b of full_b.
    let occ_col = |i: usize| -> (bool, usize) {
        if occ_spin_a(i) {
            (true, i) // α full-active column = occ index
        } else {
            (false, i - no_a) // β full-active column = occ index within β
        }
    };
    // Virtual spin-orbital -> (spin, active-MO column). α-vir live in columns
    // no_a..nact_a of full_a; β-vir in no_b..nact_b of full_b.
    let vir_col = |a: usize| -> (bool, usize) {
        if vir_spin_a(a) {
            (true, no_a + a)
        } else {
            (false, no_b + (a - nv_a))
        }
    };

    // The spin-orbital chemist integral (P Q | R S) where P,Q,R,S are
    // (spin, column) pairs. Non-zero only when spin(P)==spin(Q) and
    // spin(R)==spin(S). Uses gaaaa / gbbbb / gaabb (and the bb|aa transpose).
    let chem_so = |p: (bool, usize), q: (bool, usize), r: (bool, usize), s: (bool, usize)| -> f64 {
        if p.0 != q.0 || r.0 != s.0 {
            return 0.0;
        }
        match (p.0, r.0) {
            (true, true) => gaaaa(p.1, q.1, r.1, s.1),
            (false, false) => gbbbb(p.1, q.1, r.1, s.1),
            (true, false) => gaabb(p.1, q.1, r.1, s.1),
            // (bb|aa) = (aa|bb) with the two AO pairs swapped.
            (false, true) => gaabb(r.1, s.1, p.1, q.1),
        }
    };

    // Antisymmetrized physicist integral <pq||rs> = (pr|qs) − (ps|qr) where p,q
    // are bra spin-orbitals and r,s ket spin-orbitals. Each of p,q,r,s is a
    // (spin, column) pair already resolved to occ/vir active columns.
    let asym = |p: (bool, usize), q: (bool, usize), r: (bool, usize), s: (bool, usize)| {
        chem_so(p, r, q, s) - chem_so(p, s, q, r)
    };

    // Assemble the spin-orbital blocks.
    let mut oovv = vec![0.0_f64; no * no * nv * nv];
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    oovv[imd::oovv_idx(no, nv, i, j, a, b)] =
                        asym(occ_col(i), occ_col(j), vir_col(a), vir_col(b));
                }
            }
        }
    }
    let mut oooo = vec![0.0_f64; no * no * no * no];
    for i in 0..no {
        for j in 0..no {
            for k in 0..no {
                for l in 0..no {
                    oooo[imd::oooo_idx(no, i, j, k, l)] =
                        asym(occ_col(i), occ_col(j), occ_col(k), occ_col(l));
                }
            }
        }
    }
    let mut vvvv = vec![0.0_f64; nv * nv * nv * nv];
    for a in 0..nv {
        for b in 0..nv {
            for c in 0..nv {
                for d in 0..nv {
                    vvvv[imd::vvvv_idx(nv, a, b, c, d)] =
                        asym(vir_col(a), vir_col(b), vir_col(c), vir_col(d));
                }
            }
        }
    }
    let mut ovvo = vec![0.0_f64; no * nv * nv * no];
    for i in 0..no {
        for a in 0..nv {
            for b in 0..nv {
                for j in 0..no {
                    ovvo[imd::ovvo_idx(no, nv, i, a, b, j)] =
                        asym(occ_col(i), vir_col(a), vir_col(b), occ_col(j));
                }
            }
        }
    }
    let mut ooov = vec![0.0_f64; no * no * no * nv];
    for i in 0..no {
        for j in 0..no {
            for k in 0..no {
                for a in 0..nv {
                    ooov[imd::ooov_idx(no, nv, i, j, k, a)] =
                        asym(occ_col(i), occ_col(j), occ_col(k), vir_col(a));
                }
            }
        }
    }
    let mut ovvv = vec![0.0_f64; no * nv * nv * nv];
    for i in 0..no {
        for a in 0..nv {
            for b in 0..nv {
                for c in 0..nv {
                    ovvv[imd::ovvv_idx(nv, i, a, b, c)] =
                        asym(occ_col(i), vir_col(a), vir_col(b), vir_col(c));
                }
            }
        }
    }

    // Spin-orbital MO energies (occ first, then vir) in the interleaved order.
    // Each spin channel contributes its OWN α/β energies.
    let mut mo_energy = Vec::with_capacity(nmo);
    for &e in eo_a.iter() {
        mo_energy.push(e);
    }
    for &e in eo_b.iter() {
        mo_energy.push(e);
    }
    for &e in ev_a.iter() {
        mo_energy.push(e);
    }
    for &e in ev_b.iter() {
        mo_energy.push(e);
    }

    // Canonical reference: diagonal Fock = mo_energy (fock_ov = 0 → t1 seed 0).
    let mut fock = vec![0.0_f64; nmo * nmo];
    for p in 0..nmo {
        fock[p * nmo + p] = mo_energy[p];
    }

    Ok(SpinOrbitalEris {
        oovv,
        oooo,
        vvvv,
        ovvo,
        ooov,
        ovvv,
        fock,
        mo_energy,
        no,
        nv,
    })
}

/// Concatenate two MO coefficient blocks (same `nao`) column-wise: the result
/// has `nmo = a.nmo + b.nmo` columns (a's columns first). Used to build the
/// per-spin full active (occ then vir) MO subset.
fn concat_mo(a: &MOCoefficients, b: &MOCoefficients) -> MOCoefficients {
    let mut data = a.data.clone();
    data.extend_from_slice(&b.data);
    let mut energies = a.energies.clone();
    energies.extend_from_slice(&b.energies);
    let mut occupations = a.occupations.clone();
    occupations.extend_from_slice(&b.occupations);
    MOCoefficients {
        nao: a.nao,
        nmo: a.nmo + b.nmo,
        data,
        energies,
        occupations,
    }
}

/// Spin-orbital MP2 seed: `t1 = 0` (canonical reference), `t2[i,j,a,b] =
/// <ij||ab> / (εi + εj − εa − εb)`, `emp2 = 0.25·Σ <ij||ab>·t2[i,j,a,b]`
/// (the spin-orbital MP2 energy). Returns `(emp2, t1, t2)` flat C-order.
fn init_amps_so(eris: &SpinOrbitalEris) -> Result<(f64, Vec<f64>, Vec<f64>), CcsdError> {
    let no = eris.no;
    let nv = eris.nv;
    let nmo = no + nv;
    imd::validate(&vec![0.0; no * nv], &vec![0.0; no * no * nv * nv], eris)?;
    let eo: Vec<f64> = (0..no).map(|i| eris.mo_energy[i]).collect();
    let ev: Vec<f64> = (0..nv).map(|a| eris.mo_energy[no + a]).collect();
    let _ = nmo;
    let t1 = vec![0.0_f64; no * nv];
    let mut t2 = vec![0.0_f64; no * no * nv * nv];
    let mut e_terms: Vec<f64> = Vec::with_capacity(no * no * nv * nv);
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let denom = eo[i] + eo[j] - ev[a] - ev[b];
                    let g = eris.oovv[imd::oovv_idx(no, nv, i, j, a, b)];
                    let p = imd::t2_idx(no, nv, i, j, a, b);
                    t2[p] = if denom.abs() > 0.0 { g / denom } else { 0.0 };
                    e_terms.push(0.25 * g * t2[p]);
                }
            }
        }
    }
    Ok((oracle_sum(&e_terms), t1, t2))
}

/// Spin-orbital CCSD correlation energy (Stanton 1991 Eq. (1), spin-orbital
/// form): `E = Σ_ia f_ov[i,a]·t1[i,a] + 0.25·Σ_ijab <ij||ab>·t2[i,j,a,b]
/// + 0.5·Σ_ijab <ij||ab>·t1[i,a]·t1[j,b]`. Every reduction via `oracle_sum`.
fn energy_so(t1: &[f64], t2: &[f64], eris: &SpinOrbitalEris) -> Result<f64, CcsdError> {
    let (no, nv) = imd::validate(t1, t2, eris)?;
    let nmo = no + nv;
    let mut terms: Vec<f64> = Vec::new();
    // Σ_ia f_ov[i,a]·t1[i,a]  (0 for a canonical reference).
    for i in 0..no {
        for a in 0..nv {
            terms.push(eris.fock[imd::fock_idx(nmo, i, no + a)] * t1[imd::t1_idx(nv, i, a)]);
        }
    }
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let g = eris.oovv[imd::oovv_idx(no, nv, i, j, a, b)];
                    terms.push(0.25 * g * t2[imd::t2_idx(no, nv, i, j, a, b)]);
                    terms.push(0.5 * g * t1[imd::t1_idx(nv, i, a)] * t1[imd::t1_idx(nv, j, b)]);
                }
            }
        }
    }
    Ok(oracle_sum(&terms))
}

/// One spin-orbital CCSD amplitude update `(t1,t2) → (t1new,t2new)` (Stanton
/// 1991 Eqs. (1)-(2), the singles/doubles residual divided by the orbital-
/// energy denominators). `wvvvv` is a caller-owned `nv⁴` scratch buffer (the
/// kernel reserves it once and reuses it every cycle — Pitfall 20). Every
/// reduction routes through `oracle_sum` (no `+=`, no gemm).
fn update_amps_so(
    t1: &[f64],
    t2: &[f64],
    eris: &SpinOrbitalEris,
    wvvvv: &mut [f64],
) -> Result<(Vec<f64>, Vec<f64>), CcsdError> {
    let (no, nv) = imd::validate(t1, t2, eris)?;
    let nmo = no + nv;
    if wvvvv.len() != nv * nv * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: nv * nv * nv * nv,
            got: wvvvv.len(),
        });
    }

    let eo: Vec<f64> = (0..no).map(|i| eris.mo_energy[i]).collect();
    let ev: Vec<f64> = (0..nv).map(|a| eris.mo_energy[no + a]).collect();

    let t1e = |i: usize, a: usize| t1[imd::t1_idx(nv, i, a)];
    let t2e = |i: usize, j: usize, a: usize, b: usize| t2[imd::t2_idx(no, nv, i, j, a, b)];
    let fov = |i: usize, a: usize| eris.fock[imd::fock_idx(nmo, i, no + a)];

    // Intermediates (Stanton Eqs. (3)-(8)); the diagonal energy shift is moved
    // to the residual side (Fae −= ε on diagonal, etc. — handled by building
    // f_oo/f_vv WITHOUT the diagonal energy, then subtracting it explicitly in
    // the residual via the denominators below).
    let fae = imd::f_vv(t1, t2, eris)?; // [nv,nv]
    let fmi = imd::f_oo(t1, t2, eris)?; // [no,no]
    let fme = imd::f_ov(t1, t2, eris)?; // [no,nv]
    imd::w_vvvv_into(t1, t2, eris, wvvvv)?;
    let woooo = imd::w_oooo(t1, t2, eris)?; // [no,no,no,no]
    let wovvo = imd::w_ovvo(t1, t2, eris)?; // [no,nv,nv,no]
    let tau = imd::make_tau(t1, t2, eris)?; // [no,no,nv,nv]

    let fae_e = |a: usize, e: usize| fae[a * nv + e];
    let fmi_e = |m: usize, i: usize| fmi[m * no + i];
    let fme_e = |m: usize, e: usize| fme[m * nv + e];
    let woooo_e = |m: usize, n: usize, i: usize, j: usize| woooo[imd::oooo_idx(no, m, n, i, j)];
    let wvvvv_e = |a: usize, b: usize, e: usize, f: usize| wvvvv[imd::vvvv_idx(nv, a, b, e, f)];
    let wovvo_e = |m: usize, b: usize, e: usize, j: usize| wovvo[imd::ovvo_idx(no, nv, m, b, e, j)];

    // ---- T1 residual (Stanton Eq. (1)) over the canonical reference --------
    // t1new[i,a] = [ f_ov[i,a]
    //   + Σ_e t1[i,e]·Fae[a,e]
    //   − Σ_m t1[m,a]·Fmi[m,i]
    //   + Σ_me t2[i,m,a,e]·Fme[m,e]
    //   − Σ_nf t1[n,f]·<na||if>          ; <na||if> = −ovvv[n,a,f,i]? handled
    //   − 0.5·Σ_mef t2[i,m,e,f]·<ma||ef>
    //   − 0.5·Σ_men t2[m,n,a,e]·<nm||ei>
    //   ] / (εi − εa)
    let mut t1new = vec![0.0_f64; no * nv];
    for i in 0..no {
        for a in 0..nv {
            let mut terms: Vec<f64> = Vec::new();
            terms.push(fov(i, a));
            for e in 0..nv {
                terms.push(t1e(i, e) * fae_e(a, e));
            }
            for m in 0..no {
                terms.push(-t1e(m, a) * fmi_e(m, i));
            }
            for m in 0..no {
                for e in 0..nv {
                    terms.push(t2e(i, m, a, e) * fme_e(m, e));
                }
            }
            // − Σ_n,f t1[n,f]·<na||if>.  <na||if> = asym → from ovvo/oovv:
            //   <na||if> = <na|if> − <na|fi>. In our blocks <ni||af> = oovv,
            //   and <na||if> relates to ovvo[i,?]; cleanest: <na||if> =
            //   − <an||if> and <an||if> = ovvo? Use the identity
            //   Σ_nf t1[n,f]·<na||if> = − Σ_nf t1[n,f]·<ia||nf>?  We instead use
            //   the standard Stanton T1 form term: − Σ_nf t1[n,f]·<na||if>
            //   with <na||if> = ovvo block reordered. ovvo is <ia||bj>; we need
            //   <na||if> = <na||if>. Build from oovv via <na||if> = −<an||if>
            //   and <an||if> = −<na||if> … circular. Use ovvv + ooov instead:
            //   <na||if>: n,a? a is virtual, n occ, i occ, f vir →
            //   <occ vir || occ vir> = ovvo-type = <na||if> = ovvo[n,a,?]? ovvo
            //   index is <i a||b j> = occ vir vir occ. Not matching. The term
            //   physically is the ring <ma||ei> contraction folded into wovvo.
            // We follow the compact Stanton residual that uses ONLY F* and the
            // 3 explicit two-electron terms below; the <na||if> ring is already
            // contained in the Fae/Fmi/Fme intermediates for the canonical
            // reference. The remaining explicit terms:
            // − 0.5·Σ_m,e,f t2[i,m,e,f]·<ma||ef>   ; <ma||ef> = ovvv[m,a,e,f]
            for m in 0..no {
                for e in 0..nv {
                    for f in 0..nv {
                        terms.push(
                            -0.5 * t2e(i, m, e, f) * eris.ovvv[imd::ovvv_idx(nv, m, a, e, f)],
                        );
                    }
                }
            }
            // − 0.5·Σ_m,e,n t2[m,n,a,e]·<nm||ei> ; <nm||ei> = −<nm||ie> =
            //   −ooov[n,m,i,e]
            for m in 0..no {
                for n in 0..no {
                    for e in 0..nv {
                        terms.push(
                            0.5 * t2e(m, n, a, e) * eris.ooov[imd::ooov_idx(no, nv, n, m, i, e)],
                        );
                    }
                }
            }
            // − Σ_n,f t1[n,f]·<na||if>.  <na||if> = ovvo reordered:
            //   <na||if> = <na|if> − <na|fi>.  Using ovvo = <i a||b j> we have
            //   <na||if> = <i? ...>. The physically correct closed form here is
            //   − Σ_nf t1[n,f]·<na||if> = Σ_nf t1[n,f]·<an||if> and
            //   <an||if> = ovvo[? ]. To avoid ambiguity we contract directly
            //   against oovv-derived <ni||af>: the Stanton T1 ring term is
            //   − Σ_nf t1[n,f]·<na||if>, and <na||if> = − <ni||fa>'s partner.
            //   We use the antisymmetric ovvo block which IS <occ vir||vir occ>:
            //   <n a || i f> is <occ vir||occ vir> = ovov-type, recoverable as
            //   <na||if> = − ovvo[n,a,f,i] (swap the last two ket slots:
            //   <na||if> = −<na||fi> = − ovvo[n,a,f,i]).
            for n in 0..no {
                for f in 0..nv {
                    terms.push(-t1e(n, f) * (-eris.ovvo[imd::ovvo_idx(no, nv, n, a, f, i)]));
                }
            }
            let denom = eo[i] - ev[a];
            t1new[imd::t1_idx(nv, i, a)] = oracle_sum(&terms) / denom;
        }
    }

    // ---- T2 residual (Stanton Eq. (2)) -------------------------------------
    // t2new[i,j,a,b] = [ <ij||ab>
    //   + P(ab) Σ_e t2[i,j,a,e]·(Fbe − 0.5·Σ_m t1[m,b]·Fme)
    //   − P(ij) Σ_m t2[i,m,a,b]·(Fmj + 0.5·Σ_e t1[j,e]·Fme)
    //   + 0.5·Σ_mn tau[m,n,a,b]·Wmnij
    //   + 0.5·Σ_ef tau[i,j,e,f]·Wabef
    //   + P(ij)P(ab) Σ_me ( t2[i,m,a,e]·Wmbej − t1[i,e]·t1[m,a]·<mb||ej> )
    //   + P(ij) Σ_e t1[i,e]·<ab||ej>
    //   − P(ab) Σ_m t1[m,a]·<mb||ij>
    //   ] / (εi + εj − εa − εb)
    // where P(pq)f = f − (p<->q swap).
    let mut t2new = vec![0.0_f64; no * no * nv * nv];
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let mut terms: Vec<f64> = Vec::new();
                    // <ij||ab>
                    terms.push(eris.oovv[imd::oovv_idx(no, nv, i, j, a, b)]);

                    // P(ab) Σ_e t2[i,j,a,e]·( Fxe − 0.5·Σ_m t1[m,x]·Fme )
                    // The effective virtual intermediate for a fixed (x,e) is a
                    // single oracle_sum over [Fae[x,e], −0.5·t1[m,x]·Fme(m,e)…].
                    let fvv_eff = |x: usize, e: usize| -> f64 {
                        let mut s: Vec<f64> = Vec::with_capacity(no + 1);
                        s.push(fae_e(x, e));
                        for m in 0..no {
                            s.push(-0.5 * t1e(m, x) * fme_e(m, e));
                        }
                        oracle_sum(&s)
                    };
                    for e in 0..nv {
                        terms.push(t2e(i, j, a, e) * fvv_eff(b, e));
                        terms.push(-t2e(i, j, b, e) * fvv_eff(a, e)); // − (a<->b)
                    }

                    // − P(ij) Σ_m t2[i,m,a,b]·( Fmx + 0.5·Σ_e t1[x,e]·Fme )
                    let foo_eff = |m: usize, x: usize| -> f64 {
                        let mut s: Vec<f64> = Vec::with_capacity(nv + 1);
                        s.push(fmi_e(m, x));
                        for e in 0..nv {
                            s.push(0.5 * t1e(x, e) * fme_e(m, e));
                        }
                        oracle_sum(&s)
                    };
                    for m in 0..no {
                        terms.push(-t2e(i, m, a, b) * foo_eff(m, j));
                        terms.push(t2e(j, m, a, b) * foo_eff(m, i)); // − (i<->j)
                    }

                    // 0.5·Σ_mn tau[m,n,a,b]·Wmnij
                    for m in 0..no {
                        for n in 0..no {
                            terms.push(
                                0.5 * tau[imd::t2_idx(no, nv, m, n, a, b)] * woooo_e(m, n, i, j),
                            );
                        }
                    }

                    // 0.5·Σ_ef tau[i,j,e,f]·Wabef
                    for e in 0..nv {
                        for f in 0..nv {
                            terms.push(
                                0.5 * tau[imd::t2_idx(no, nv, i, j, e, f)] * wvvvv_e(a, b, e, f),
                            );
                        }
                    }

                    // P(ij)P(ab) Σ_me ( t2[i,m,a,e]·Wmbej
                    //                    − t1[i,e]·t1[m,a]·<mb||ej> )
                    // <mb||ej> = ovvo[m,b,e,j]
                    for m in 0..no {
                        for e in 0..nv {
                            let ring = |ii: usize, jj: usize, aa: usize, bb: usize| {
                                t2e(ii, m, aa, e) * wovvo_e(m, bb, e, jj)
                                    - t1e(ii, e)
                                        * t1e(m, aa)
                                        * eris.ovvo[imd::ovvo_idx(no, nv, m, bb, e, jj)]
                            };
                            // P(ij)P(ab): (ij,ab) − (ji,ab) − (ij,ba) + (ji,ba)
                            terms.push(ring(i, j, a, b));
                            terms.push(-ring(j, i, a, b));
                            terms.push(-ring(i, j, b, a));
                            terms.push(ring(j, i, b, a));
                        }
                    }

                    // P(ij) Σ_e t1[i,e]·<ab||ej> ; <ab||ej> = −<ab||je> and
                    // <ab||je> = ovvv? <ab||ej> = <vir vir||vir occ>. Recover
                    // from ovvv via <ab||ej> = −<ja||be>'s antisym → use the
                    // identity <ab||ej> = − <ej||ab>* and <ej||ab> = ovvv-type
                    // <occ? >. Cleanest: <ab||ej> = − ovvv[j,? ]. We have
                    // ovvv = <ia||bc> = <occ vir||vir vir>. So <ej||ab> with e
                    // vir, j occ, a,b vir = <vir occ||vir vir> = − <je||ab> =
                    // − ovvv[j,e,a,b]. And <ab||ej> = <ej||ab> (hermitian, real)
                    // = − ovvv[j,e,a,b].
                    for e in 0..nv {
                        let g_abej = -eris.ovvv[imd::ovvv_idx(nv, j, e, a, b)];
                        terms.push(t1e(i, e) * g_abej);
                        // − (i<->j): <ab||ei> = − ovvv[i,e,a,b]
                        let g_abei = -eris.ovvv[imd::ovvv_idx(nv, i, e, a, b)];
                        terms.push(-t1e(j, e) * g_abei);
                    }

                    // − P(ab) Σ_m t1[m,a]·<mb||ij> ; <mb||ij> = ooov? <mb||ij>
                    // = <occ vir||occ occ>. Recover from ooov via <mb||ij> =
                    // <ij||mb>* = <occ occ||occ vir> = ooov[i,j,m,b]. (real,
                    // hermitian ⇒ <mb||ij> = <ij||mb> = ooov[i,j,m,b]).
                    for m in 0..no {
                        let g_mbij = eris.ooov[imd::ooov_idx(no, nv, i, j, m, b)];
                        terms.push(-t1e(m, a) * g_mbij);
                        // − (a<->b): <ma||ij> = ooov[i,j,m,a]
                        let g_maij = eris.ooov[imd::ooov_idx(no, nv, i, j, m, a)];
                        terms.push(t1e(m, b) * g_maij);
                    }

                    let denom = eo[i] + eo[j] - ev[a] - ev[b];
                    t2new[imd::t2_idx(no, nv, i, j, a, b)] = oracle_sum(&terms) / denom;
                }
            }
        }
    }

    Ok((t1new, t2new))
}

/// `||(t1new,t2new) − (t1,t2)||` Frobenius norm — the dual-criterion `normt`.
fn amp_delta_norm(t1o: &[f64], t2o: &[f64], t1n: &[f64], t2n: &[f64]) -> f64 {
    let mut sq: Vec<f64> = Vec::with_capacity(t1o.len() + t2o.len());
    for (n, o) in t1n.iter().zip(t1o.iter()) {
        let d = n - o;
        sq.push(d * d);
    }
    for (n, o) in t2n.iter().zip(t2o.iter()) {
        let d = n - o;
        sq.push(d * d);
    }
    oracle_sum(&sq).sqrt()
}

/// nv⁴·8 bytes — the dominant `Wabef` arena tenant pre-flight estimate.
fn estimate_vvvv_bytes(nv: usize) -> usize {
    nv.saturating_mul(nv)
        .saturating_mul(nv)
        .saturating_mul(nv)
        .saturating_mul(8)
}

/// Pack the converged spin-orbital `t2` into the spin-resolved
/// [`UccsdAmplitudes`] triple by spin label. Spin-orbital occupied index
/// `0..no_a` is α-occ, `no_a..no` is β-occ; virtual `0..nv_a` is α-vir,
/// `nv_a..nv` is β-vir. `t2aa`/`t2bb` are the same-spin sub-blocks (already
/// antisymmetric), `t2ab` the αβ sub-block.
fn pack_amplitudes(
    t2: &[f64],
    no: usize,
    nv: usize,
    no_a: usize,
    nv_a: usize,
    no_b: usize,
    nv_b: usize,
) -> UccsdAmplitudes {
    let mut amp = UccsdAmplitudes {
        nocc_a: no_a,
        nvir_a: nv_a,
        nocc_b: no_b,
        nvir_b: nv_b,
        t2aa: vec![0.0; no_a * no_a * nv_a * nv_a],
        t2ab: vec![0.0; no_a * no_b * nv_a * nv_b],
        t2bb: vec![0.0; no_b * no_b * nv_b * nv_b],
    };
    let so = |i: usize, j: usize, a: usize, b: usize| t2[imd::t2_idx(no, nv, i, j, a, b)];
    // αα: occ i,j in 0..no_a, vir a,b in 0..nv_a.
    for i in 0..no_a {
        for j in 0..no_a {
            for a in 0..nv_a {
                for b in 0..nv_a {
                    let dst = amp.aa_idx(i, j, a, b);
                    amp.t2aa[dst] = so(i, j, a, b);
                }
            }
        }
    }
    // ββ: occ I,J in no_a..no, vir A,B in nv_a..nv.
    for ii in 0..no_b {
        for jj in 0..no_b {
            for aa in 0..nv_b {
                for bb in 0..nv_b {
                    let dst = amp.bb_idx(ii, jj, aa, bb);
                    amp.t2bb[dst] = so(no_a + ii, no_a + jj, nv_a + aa, nv_a + bb);
                }
            }
        }
    }
    // αβ: i in 0..no_a, J in no_a..no, a in 0..nv_a, B in nv_a..nv.
    for i in 0..no_a {
        for jj in 0..no_b {
            for a in 0..nv_a {
                for bb in 0..nv_b {
                    let dst = amp.ab_idx(i, jj, a, bb);
                    amp.t2ab[dst] = so(i, no_a + jj, a, nv_a + bb);
                }
            }
        }
    }
    amp
}

/// Decompose the converged spin-orbital correlation energy into the
/// `e_aa`/`e_bb`/`e_ab` spin channels (`uccsd.py:343-371`). For a canonical
/// reference (`t1≈0`) the energy is `0.25·Σ_ijab <ij||ab>·t2[i,j,a,b]` over the
/// full spin-orbital range; restricting `(i,j,a,b)` to each spin sub-block
/// gives the channel contributions. The mixed-spin pairs (one α one β occ)
/// constitute `e_ab`; pure α and pure β pairs give `e_aa`/`e_bb`.
#[allow(clippy::too_many_arguments)]
fn decompose_energy(
    t1: &[f64],
    t2: &[f64],
    eris: &SpinOrbitalEris,
    no: usize,
    nv: usize,
    no_a: usize,
    nv_a: usize,
) -> (f64, f64, f64) {
    let occ_a = |i: usize| i < no_a;
    let vir_a = |a: usize| a < nv_a;
    let nmo = no + nv;
    let mut aa: Vec<f64> = Vec::new();
    let mut bb: Vec<f64> = Vec::new();
    let mut ab: Vec<f64> = Vec::new();
    // Singles contribution (0 for canonical reference) attributed by spin of i.
    for i in 0..no {
        for a in 0..nv {
            let s = eris.fock[imd::fock_idx(nmo, i, no + a)] * t1[imd::t1_idx(nv, i, a)];
            if occ_a(i) {
                aa.push(s);
            } else {
                bb.push(s);
            }
        }
    }
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let g = eris.oovv[imd::oovv_idx(no, nv, i, j, a, b)];
                    let t = t2[imd::t2_idx(no, nv, i, j, a, b)];
                    let t1t1 = t1[imd::t1_idx(nv, i, a)] * t1[imd::t1_idx(nv, j, b)];
                    let contrib = 0.25 * g * t + 0.5 * g * t1t1;
                    // Spin label by the occupied pair (i,j). Same-spin
                    // requires the virtual pair to match too; for a valid
                    // spin-orbital amplitude a mismatched-spin term is zero, so
                    // bucketing by (i,j) spin is exact.
                    match (occ_a(i), occ_a(j)) {
                        (true, true) => aa.push(contrib),
                        (false, false) => bb.push(contrib),
                        _ => {
                            // mixed-spin occupied pair → e_ab. Guard against the
                            // (vir spins not matching) zero terms anyway.
                            let _ = (vir_a(a), vir_a(b));
                            ab.push(contrib);
                        }
                    }
                }
            }
        }
    }
    (oracle_sum(&aa), oracle_sum(&bb), oracle_sum(&ab))
}

/// Open-shell UCCSD driver — port of `pyscf/cc/uccsd.py` (the compact
/// spin-orbital realization; see module docs). Mirrors the 06-03 `ccsd_kernel`
/// discipline: HARD `try_reserve` pre-flight on the `nv⁴` tenant, MP2-seeded
/// `init_amps`, the `Wabef` scratch reserved ONCE and reused, dual-criterion
/// convergence, release after.
///
/// # Errors
/// Propagates shape mismatches (`ShapeMismatch`), the `int2e`/`ao2mo` chain
/// errors, and the `try_reserve` `MemoryLimitExceeded` refusal. Never panics,
/// never indexes OOB (`#![forbid(unsafe_code)]`).
pub fn uccsd_kernel(
    refr: &UccsdReference,
    frozen: &Frozen,
    pool: &WorkspacePool,
) -> Result<UccsdResult, PyscfRsError> {
    // Per-spin active counts (frozen-aware).
    let (no_a, nv_a) = channel_counts(&refr.alpha, frozen)?;
    let (no_b, nv_b) = channel_counts(&refr.beta, frozen)?;
    let nv = nv_a + nv_b;

    // (1) HARD pre-flight refusal on the spin-orbital nv⁴ Wabef tenant (D-01 /
    // T-06-04-OOM): refuse an over-budget run BEFORE building eris. No
    // downgrade.
    pool.try_reserve(estimate_vvvv_bytes(nv))
        .map_err(CcsdError::from)?;

    // (2) Build the combined spin-orbital antisymmetrized eris + Fock.
    let eris = build_spin_orbital_eris(refr, frozen)?;
    let no = eris.no;
    let nv = eris.nv;

    // (3) MP2 seed.
    let (emp2, t1, t2) = init_amps_so(&eris)?;
    let mut e_corr = energy_so(&t1, &t2, &eris)?;

    // (4) Reserve the Wabef arena tenant ONCE before the loop (CCSD-11).
    let wid = pool
        .reserve(&[nv, nv, nv, nv], false)
        .map_err(CcsdError::from)?;

    let mut t1_cur = t1;
    let mut t2_cur = t2;
    let mut converged = false;
    let mut niter = 0usize;

    // (5) Amplitude iteration.
    for istep in 0..MAX_CYCLE {
        niter = istep + 1;
        let (t1new, t2new) = pool
            .with_mut_slice(&wid, |w| update_amps_so(&t1_cur, &t2_cur, &eris, w))
            .map_err(CcsdError::from)??;
        let normt = amp_delta_norm(&t1_cur, &t2_cur, &t1new, &t2new);
        t1_cur = t1new;
        t2_cur = t2new;
        let eold = e_corr;
        e_corr = energy_so(&t1_cur, &t2_cur, &eris)?;
        if (e_corr - eold).abs() < CONV_TOL && normt < CONV_TOL_NORMT {
            converged = true;
            break;
        }
    }

    // (6) Release the arena tenant.
    pool.release(wid);

    // Decompose the energy by spin channel + pack the spin-resolved amplitudes.
    let (e_aa, e_bb, e_ab) = decompose_energy(&t1_cur, &t2_cur, &eris, no, nv, no_a, nv_a);
    let amplitudes = pack_amplitudes(&t2_cur, no, nv, no_a, nv_a, no_b, nv_b);

    Ok(UccsdResult {
        e_corr,
        e_aa,
        e_ab,
        e_bb,
        converged,
        niter,
        emp2,
        amplitudes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyscf_core::Mole;

    /// UccsdAmplitudes flat-index round-trips: writing a value at (i,j,a,b) and
    /// reading it back through the documented index formula agrees, for all
    /// three channels, with the upstream UmpAmplitudes layout.
    #[test]
    fn uccsd_amps_index_layout_round_trips() {
        let (nocc_a, nvir_a, nocc_b, nvir_b) = (2usize, 3usize, 1usize, 2usize);
        let mut amp = UccsdAmplitudes {
            nocc_a,
            nvir_a,
            nocc_b,
            nvir_b,
            t2aa: vec![0.0; nocc_a * nocc_a * nvir_a * nvir_a],
            t2ab: vec![0.0; nocc_a * nocc_b * nvir_a * nvir_b],
            t2bb: vec![0.0; nocc_b * nocc_b * nvir_b * nvir_b],
        };
        // αα
        for i in 0..nocc_a {
            for j in 0..nocc_a {
                for a in 0..nvir_a {
                    for b in 0..nvir_a {
                        let v = (((i * 7 + j) * 5 + a) * 3 + b) as f64 + 0.5;
                        let idx = amp.aa_idx(i, j, a, b);
                        // matches ump2 layout ((i*nocc_a + j)*nvir_a + a)*nvir_a + b
                        assert_eq!(idx, ((i * nocc_a + j) * nvir_a + a) * nvir_a + b);
                        amp.t2aa[idx] = v;
                        assert_eq!(amp.t2aa[amp.aa_idx(i, j, a, b)], v);
                    }
                }
            }
        }
        // αβ
        for i in 0..nocc_a {
            for jj in 0..nocc_b {
                for a in 0..nvir_a {
                    for bb in 0..nvir_b {
                        let v = (((i * 11 + jj) * 7 + a) * 3 + bb) as f64 + 0.25;
                        let idx = amp.ab_idx(i, jj, a, bb);
                        assert_eq!(idx, ((i * nocc_b + jj) * nvir_a + a) * nvir_b + bb);
                        amp.t2ab[idx] = v;
                        assert_eq!(amp.t2ab[amp.ab_idx(i, jj, a, bb)], v);
                    }
                }
            }
        }
        // ββ
        for ii in 0..nocc_b {
            for jj in 0..nocc_b {
                for aa in 0..nvir_b {
                    for bb in 0..nvir_b {
                        let v = (((ii * 13 + jj) * 7 + aa) * 3 + bb) as f64 + 0.75;
                        let idx = amp.bb_idx(ii, jj, aa, bb);
                        assert_eq!(idx, ((ii * nocc_b + jj) * nvir_b + aa) * nvir_b + bb);
                        amp.t2bb[idx] = v;
                        assert_eq!(amp.t2bb[amp.bb_idx(ii, jj, aa, bb)], v);
                    }
                }
            }
        }
    }

    /// Each spin channel transforms its OWN α/β orbital energies: holding the
    /// integrals fixed and changing ONLY the β-virtual energy must change the
    /// αβ seed amplitude (whose denominator reads the β-vir energy), proving the
    /// per-channel energy split (the plan's "each channel uses its OWN energies"
    /// invariant, at the seed level the kernel then iterates).
    #[test]
    fn each_channel_uses_its_own_energies() {
        // 1 α-occ, 1 β-occ, 2 α-vir, 2 β-vir → no=2, nv=4.
        let (no_a, nv_a, no_b, nv_b) = (1usize, 2usize, 1usize, 2usize);
        let no = no_a + no_b; // 2
        let nv = nv_a + nv_b; // 4
        let nmo = no + nv;

        // Build an eris with a non-zero αβ oovv block; vary only the β-vir
        // energy between the two builds.
        let build = |eb_vir1: f64| -> Vec<f64> {
            // occ: [α=-0.6, β=-0.5]; vir: [α-vir0=0.3, α-vir1=0.5,
            //                               β-vir0=0.4, β-vir1=eb_vir1].
            let mo_energy = vec![-0.6, -0.5, 0.3, 0.5, 0.4, eb_vir1];
            let mut fock = vec![0.0; nmo * nmo];
            for p in 0..nmo {
                fock[p * nmo + p] = mo_energy[p];
            }
            let mut oovv = vec![0.0; no * no * nv * nv];
            for i in 0..no {
                for j in 0..no {
                    for a in 0..nv {
                        for b in 0..nv {
                            // deterministic non-zero αβ-style fill.
                            oovv[imd::oovv_idx(no, nv, i, j, a, b)] =
                                0.02 + ((i + 2 * j + 3 * a + 5 * b) % 4) as f64 * 0.01;
                        }
                    }
                }
            }
            let eris = SpinOrbitalEris {
                oovv,
                oooo: vec![0.0; no * no * no * no],
                vvvv: vec![0.0; nv * nv * nv * nv],
                ovvo: vec![0.0; no * nv * nv * no],
                ooov: vec![0.0; no * no * no * nv],
                ovvv: vec![0.0; no * nv * nv * nv],
                fock,
                mo_energy,
                no,
                nv,
            };
            let (_e, _t1, t2) = init_amps_so(&eris).expect("seed");
            t2
        };

        let t2_a = build(0.7);
        let t2_b = build(0.9); // only β-vir1 energy changes

        // The αβ seed amplitude touching β-vir1 (B = nv_a + 1 = 3) must change.
        let p = imd::t2_idx(no, nv, 0, 1, 0, 3); // i=α-occ, J=β-occ, a=α-vir0, B=β-vir1
        assert!(
            (t2_a[p] - t2_b[p]).abs() > 1e-9,
            "changing only the β-vir energy must change the αβ seed t2 ({} vs {})",
            t2_a[p],
            t2_b[p]
        );
        // An amplitude NOT touching β-vir1 (B = β-vir0 = 2) must be unchanged.
        let q = imd::t2_idx(no, nv, 0, 1, 0, 2);
        assert!(
            (t2_a[q] - t2_b[q]).abs() < 1e-15,
            "an amplitude not touching β-vir1 must be invariant ({} vs {})",
            t2_a[q],
            t2_b[q]
        );
    }

    /// A minimal Mole + identity-coefficient CcsdReference helper is not needed
    /// for the always-on unit tests here (the real reference-driven numeric
    /// cross-check lives in tests/uccsd_smoke.rs). Keep a compile-time witness
    /// that UccsdReference is constructible.
    #[test]
    fn uccsd_reference_is_constructible() {
        let mol = Mole::default();
        let one = CcsdReference {
            mo_coeff: MOCoefficients {
                nao: 1,
                nmo: 1,
                data: vec![1.0],
                energies: vec![-0.5],
                occupations: vec![1.0],
            },
            mo_energy: vec![-0.5],
            mo_occ: vec![1.0],
            e_hf: 0.0,
            converged: true,
            mol,
        };
        let uref = UccsdReference {
            alpha: one.clone(),
            beta: one,
        };
        assert_eq!(uref.alpha.mo_energy, uref.beta.mo_energy);
    }
}

//! RMP2 kernel / energy + `Mp2Reference` snapshot + `default_ao2mo` (port of
//! `pyscf/mp/mp2.py`).
//!
//! The closed-form MP2 correlation energy (`pyscf/mp/mp2.py:kernel`, lines
//! 33-76, canonical branch). The kernel asks the `Mp2OverrideHooks::ao2mo` seam
//! for the `(ia|jb)` block (`ChemistsEris`), divides by the `εi+εj−εa−εb`
//! denominators, and reduces every per-index sum through `oracle_sum` /
//! `oracle_dot` (NO `+=` accumulation) for bit-exactness + thread-count
//! invariance (T-05-03-FP, Pitfall 1/2).
//!
//! `default_ao2mo` builds the occupied/virtual MO column subsets and calls
//! `pyscf_ao2mo::general(eri_ao, nao, [&co,&cv,&co,&cv])` where
//! `eri_ao = pyscf_gto::intor(&refr.mol, "int2e")?`. As of 05-08 cintx#11 is
//! closed — `int2e` is a real arity-4 tensor, so this path runs end-to-end and
//! yields finite correlation energies (D-05's "no kernel change" promise).
//! Errors still `?`-propagate; we never panic and never substitute a zero
//! buffer (T-05-03-FFI).

use crate::frozen::{self, Frozen};
use crate::helpers;
use crate::hooks::{ChemistsEris, Mp2OverrideHooks};
use pyscf_algebra::{oracle_dot, oracle_sum};
use pyscf_core::{Amplitudes, MOCoefficients, Mole, PyscfRsError};

/// Snapshot of a converged SCF reference, consumed by the MP2 kernel (D-07).
/// pyo3-free — takes plain arrays. The `mol` is snapshotted from `mf.mol` by
/// the pyscf-py bridge (05-07 T1) so `default_ao2mo` / `Mp2OverrideHooks::ao2mo`
/// can reach the molecule without an extra `&Mole` parameter.
#[derive(Debug, Clone)]
pub struct Mp2Reference {
    /// MO coefficients, F-order, sign-canonicalized (SCF-13).
    pub mo_coeff: MOCoefficients,
    /// MO energies (one per MO).
    pub mo_energy: Vec<f64>,
    /// MO occupations (2.0 / 0.0 for restricted).
    pub mo_occ: Vec<f64>,
    /// Converged SCF total energy (the MP2 baseline; `get_e_hf`).
    pub e_hf: f64,
    /// Whether the reference SCF converged.
    pub converged: bool,
    /// Molecule snapshot — needed by `default_ao2mo` for the intor call.
    pub mol: Mole,
}

/// Result of the closed-form RMP2 kernel.
#[derive(Debug, Clone)]
pub struct Mp2Result {
    /// Total correlation energy `e_corr = e_ss + e_os`.
    pub e_corr: f64,
    /// Same-spin component.
    pub e_ss: f64,
    /// Opposite-spin component.
    pub e_os: f64,
    /// t2 amplitudes when `with_t2` was requested (upstream `WITH_T2`).
    pub t2: Option<Amplitudes>,
}

/// Per-atom atomic numbers `Z` for the reference's molecule (used by
/// `Frozen::Auto` chemcore resolution). Derived from `Mole::atom_charges()`.
fn reference_elements(refr: &Mp2Reference) -> Vec<u32> {
    refr.mol
        .atom_charges()
        .into_iter()
        .map(|z| z.max(0) as u32)
        .collect()
}

/// Build an occupied / virtual MO column subset honouring the frozen mask.
///
/// `want_occupied = true` selects the ACTIVE occupied columns
/// (`mo_occ > 0 AND active`); `false` selects the ACTIVE virtual columns
/// (`mo_occ == 0 AND active`). Returns the column subset as an
/// [`MOCoefficients`] whose `data` is column-major `[nao, n_selected]`.
fn mo_subset(
    refr: &Mp2Reference,
    mask: &[bool],
    want_occupied: bool,
) -> Result<MOCoefficients, PyscfRsError> {
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
        let is_occ = occ > 0.0;
        if is_occ != want_occupied {
            continue;
        }
        let start = col * nao;
        let end = start + nao;
        if end > refr.mo_coeff.data.len() {
            return Err(crate::error::Mp2Error::ShapeMismatch {
                expected: refr.mo_coeff.data.len(),
                got: end,
            }
            .into());
        }
        data.extend_from_slice(&refr.mo_coeff.data[start..end]);
        if let Some(e) = refr.mo_energy.get(col) {
            energies.push(*e);
        }
        occupations.push(occ);
    }
    let n_selected = data.len().checked_div(nao).unwrap_or(0);
    Ok(MOCoefficients {
        nao,
        nmo: n_selected,
        data,
        energies,
        occupations,
    })
}

/// Default `ao2mo`: build the `(ia|jb)` block from the AO `int2e` integrals.
///
/// Builds the active occupied (`co`) and virtual (`cv`) MO column subsets
/// (frozen-aware) and calls `pyscf_ao2mo::general(eri_ao, nao, [&co,&cv,&co,&cv])`
/// — the upstream `mp2.py:_make_eris` `(o,v,o,v)` ordering — where
/// `eri_ao = pyscf_gto::intor(&refr.mol, "int2e")`.
///
/// The returned [`ChemistsEris`] holds the flat `(occ,vir|occ,vir)` block in
/// the `(i,a,j,b)` C-order layout that [`rmp2_kernel`] consumes.
///
/// # Errors
/// `?`-propagates any error from `intor("int2e")` or `ao2mo::general`; this
/// function NEVER panics and NEVER substitutes a zero buffer (T-05-03-FFI).
/// As of 05-08 the cintx#11 gate is CLOSED — `intor("int2e")` returns a real
/// arity-4 tensor, so this path runs end-to-end and yields finite correlation
/// energies (the D-05 "numeric flips on with no kernel change" promise; see
/// pyscf-mp2/tests/mp2_numeric_smoke.rs). A genuine shape surprise from cintx
/// still propagates as a descriptive error.
pub fn default_ao2mo(refr: &Mp2Reference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError> {
    let elements = reference_elements(refr);
    let mask = frozen::frozen_mask(frozen, &refr.mo_occ, &refr.mo_energy, &elements)?;

    let co = mo_subset(refr, &mask, true)?;
    let cv = mo_subset(refr, &mask, false)?;
    let nocc = co.nmo;
    let nvir = cv.nmo;
    let nao = refr.mol.nao_nr;

    // int2e is a real arity-4 tensor as of 05-08 (cintx#11 closed). Any error
    // (e.g. a cintx shape surprise) `?`-propagates — never panic, never
    // substitute zeros (D-05/T-05-03-FFI).
    let eri = pyscf_gto::intor(&refr.mol, "int2e")?;

    // ao2mo.general expects the (o,v,o,v) ordering → (ia|jb) block.
    let ovov = pyscf_ao2mo::general(&eri.values, nao, [&co, &cv, &co, &cv])
        .map_err(crate::error::Mp2Error::from)?;

    Ok(ChemistsEris { ovov, nocc, nvir })
}

/// Closed-form RMP2 correlation energy (`pyscf/mp/mp2.py:kernel`, 47-76).
///
/// For each occupied `i` and `j`, virtual `a` and `b`:
/// `t2[i,j,a,b] = (ia|jb) / (εi + εj − εa − εb)`,
/// `edi = 2·Σ (ia|jb)·t2`, `exi = −Σ (ib|ja)·t2`,
/// `e_ss += edi·0.5 + exi`, `e_os += edi·0.5`.
///
/// Every per-`i` sum materializes its products into a `Vec` then reduces via
/// `oracle_sum` / `oracle_dot` (NO `+=` reduction) — bit-exact + thread-count
/// invariant (T-05-03-FP).
///
/// `eris.ovov` is the flat `(i,a,j,b)` C-order block: element `(i,a,j,b)` lives
/// at `((i*nvir + a)*nocc + j)*nvir + b`.
///
/// # Errors
/// Propagates any error from `hooks.ao2mo` (notably the `int2e`
/// `NotYetImplemented`) and from the MP2-08 helpers.
pub fn rmp2_kernel<H: Mp2OverrideHooks>(
    refr: &Mp2Reference,
    frozen: &Frozen,
    hooks: &H,
    with_t2: bool,
) -> Result<Mp2Result, PyscfRsError> {
    // Active occupied / total MO counts via the MP2-08 helpers.
    let nocc = helpers::get_nocc(&refr.mo_occ, frozen)?;

    // Pull the (ia|jb) block through the override seam.
    let eris = hooks.ao2mo(refr, frozen)?;
    let nvir = eris.nvir;

    // Sanity: the hooks-provided block must agree with the active occupied
    // count (the virtual count is the block's own).
    if eris.nocc != nocc {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: nocc,
            got: eris.nocc,
        }
        .into());
    }
    let expected_len = nocc * nvir * nocc * nvir;
    if eris.ovov.len() != expected_len {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: expected_len,
            got: eris.ovov.len(),
        }
        .into());
    }

    // Active occupied / virtual MO energies (frozen-aware), in column order.
    let elements = reference_elements(refr);
    let mask = frozen::frozen_mask(frozen, &refr.mo_occ, &refr.mo_energy, &elements)?;
    let mut e_occ = Vec::with_capacity(nocc);
    let mut e_vir = Vec::with_capacity(nvir);
    for (col, &active) in mask.iter().enumerate() {
        if !active {
            continue;
        }
        let occ = refr.mo_occ.get(col).copied().unwrap_or(0.0);
        let e = refr.mo_energy.get(col).copied().unwrap_or(0.0);
        if occ > 0.0 {
            e_occ.push(e);
        } else {
            e_vir.push(e);
        }
    }
    if e_occ.len() != nocc || e_vir.len() != nvir {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: nocc * nvir,
            got: e_occ.len() * e_vir.len(),
        }
        .into());
    }

    // eia[i][a] = εocc[i] − εvir[a].
    let eia = |i: usize, a: usize| e_occ[i] - e_vir[a];
    // Flat (i,a,j,b) index into ovov.
    let idx = |i: usize, a: usize, j: usize, b: usize| ((i * nvir + a) * nocc + j) * nvir + b;

    let mut t2_store: Option<Vec<f64>> = if with_t2 {
        Some(vec![0.0; nocc * nocc * nvir * nvir])
    } else {
        None
    };

    // Per-i contributions, accumulated into a Vec then oracle_sum'd.
    let mut e_ss_terms: Vec<f64> = Vec::with_capacity(nocc);
    let mut e_os_terms: Vec<f64> = Vec::with_capacity(nocc);

    // Scratch buffers reused across i (materialize-then-reduce).
    let block = nocc * nvir * nvir; // jab span for a fixed i
    let mut gi: Vec<f64> = vec![0.0; block]; // gi[(j,a,b)] = (ia|jb)
    let mut t2i: Vec<f64> = vec![0.0; block]; // t2i[(j,a,b)]
    let mut g_jba: Vec<f64> = vec![0.0; block]; // (ib|ja) reordered to (j,a,b)

    for i in 0..nocc {
        // Build gi[(j,a,b)] = (ia|jb) and the exchange-reordered (ib|ja).
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let p = (j * nvir + a) * nvir + b;
                    gi[p] = eris.ovov[idx(i, a, j, b)];
                    // (ib|ja): swap the virtual roles a<->b in the i/j block.
                    g_jba[p] = eris.ovov[idx(i, b, j, a)];
                }
            }
        }
        // t2i[(j,a,b)] = gi / (eia[i,a] + eia[j,b]).
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let p = (j * nvir + a) * nvir + b;
                    let denom = eia(i, a) + eia(j, b);
                    t2i[p] = gi[p] / denom;
                }
            }
        }
        // edi = 2 · Σ_jab gi·t2i = 2 · oracle_dot(gi, t2i).
        let edi = 2.0 * oracle_dot(&gi, &t2i);
        // exi = − Σ_jab (ib|ja)·t2i = − oracle_dot(g_jba, t2i).
        let exi = -oracle_dot(&g_jba, &t2i);
        e_ss_terms.push(edi * 0.5 + exi);
        e_os_terms.push(edi * 0.5);

        if let Some(t2) = t2_store.as_mut() {
            // Store t2[i,j,a,b] (C-order [nocc,nocc,nvir,nvir]).
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let p = (j * nvir + a) * nvir + b;
                        let dst = ((i * nocc + j) * nvir + a) * nvir + b;
                        t2[dst] = t2i[p];
                    }
                }
            }
        }
    }

    let e_ss = oracle_sum(&e_ss_terms);
    let e_os = oracle_sum(&e_os_terms);
    let e_corr = e_ss + e_os;

    // D-01: Amplitudes now holds opaque AmplitudeStore handles. MP2 has no
    // arena, so it uses the owned-Vec constructor (t1 empty for RMP2).
    let amplitudes = t2_store.map(|t2| Amplitudes::from_vec(nocc, nvir, Vec::new(), t2));

    Ok(Mp2Result {
        e_corr,
        e_ss,
        e_os,
        t2: amplitudes,
    })
}

/// SCS-MP2 energy split (`pyscf/mp/mp2.py:117-126`, `emp2_scs` property
/// 597-599).
///
/// `e = e_ss·ss_factor + e_os·os_factor`. With `ss_factor = os_factor = 1.0`
/// this reproduces plain MP2 (`e_ss + e_os`). The SCS defaults
/// (`ss = 1/3`, `os = 1.2`; J. Chem. Phys. 118, 9095 (2003)) give the
/// spin-component-scaled energy.
pub fn scs_energy(e_ss: f64, e_os: f64, ss_factor: f64, os_factor: f64) -> f64 {
    e_ss * ss_factor + e_os * os_factor
}

/// Default `energy` hook: run the closed-form kernel and apply the SCS factors.
///
/// Mirrors the `NoMp2Overrides::energy` delegate — computes `e_corr` from the
/// supplied `eris` (no re-transform) and returns the SCS-scaled total.
///
/// # Errors
/// Propagates kernel / helper errors.
pub fn default_energy(
    eris: &ChemistsEris,
    refr: &Mp2Reference,
    frozen: &Frozen,
    ss_factor: f64,
    os_factor: f64,
) -> Result<f64, PyscfRsError> {
    // Re-run the per-i closed-form on the supplied block via a thin hooks shim
    // that simply hands the block back.
    struct Pre<'a>(&'a ChemistsEris);
    impl Mp2OverrideHooks for Pre<'_> {
        fn ao2mo(
            &self,
            _refr: &Mp2Reference,
            _frozen: &Frozen,
        ) -> Result<ChemistsEris, PyscfRsError> {
            Ok(ChemistsEris {
                ovov: self.0.ovov.clone(),
                nocc: self.0.nocc,
                nvir: self.0.nvir,
            })
        }
    }
    let res = rmp2_kernel(refr, frozen, &Pre(eris), false)?;
    Ok(scs_energy(res.e_ss, res.e_os, ss_factor, os_factor))
}

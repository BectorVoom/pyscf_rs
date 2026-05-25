//! UMP2 spin-block kernel / energy (port `pyscf/mp/ump2.py`, lines 35-109).
//!
//! The open-shell (unrestricted) MP2 correlation energy is the sum of three
//! spin-channel contributions:
//!
//! - `e_aa` (same-spin αα): antisymmetrized — `0.5·Σ t2·(ia|jb) − 0.5·Σ t2·(ib|ja)`
//! - `e_bb` (same-spin ββ): identical form over the β orbitals/integrals
//! - `e_ab` (opposite-spin αβ): direct only — `Σ t2·(ia|JB)` (no exchange term)
//!
//! `e_corr = e_aa + e_bb + e_ab` (`ump2.py:107`). Each spin channel transforms
//! its OWN α/β orbital energies (`ump2.py:50-52`): the αα/αβ channels use the α
//! occupied energies for `i`/`j`, the ββ channel the β energies, and the αβ
//! channel mixes α (for `i`,`a`) with β (for `J`,`B`).
//!
//! There is NO in-repo analog for spin-resolved amplitudes — the single-channel
//! [`pyscf_core::Amplitudes`] holds one `t2`. Phase 5 ADDS the
//! [`UmpAmplitudes`] triple (`t2aa`/`t2ab`/`t2bb`), mirroring the upstream
//! `(t2aa, t2ab, t2bb)` tuple (`ump2.py:59`).
//!
//! EVERY per-channel reduction materializes its products into a `Vec` then
//! reduces via [`oracle_dot`]/[`oracle_sum`] (NO `+=` accumulation) — bit-exact
//! and thread-count invariant (T-05-04-FP, Pitfall 1/2).
//!
//! ## Synthetic-block test path
//! [`ump2_kernel`] takes the three [`ChemistsEris`] blocks DIRECTLY (already
//! transformed). This is the always-on synthetic-block path the structural
//! tests exercise (no `intor`): a caller (the pyscf-py bridge in 05-07, or
//! `default_ao2mo` once arity-4 `int2e` lands) builds `eris_aa`/`eris_ab`/
//! `eris_bb` per channel and feeds them in.

use crate::frozen::{self, Frozen};
use crate::helpers;
use crate::hooks::ChemistsEris;
use crate::mp2::Mp2Reference;
use pyscf_algebra::{oracle_dot, oracle_sum};
use pyscf_core::PyscfRsError;

/// Spin-resolved MP2 amplitude triple (`ump2.py:56-59`'s `(t2aa, t2ab, t2bb)`).
///
/// No in-repo analog: the single-channel [`pyscf_core::Amplitudes`] holds one
/// `t2`. This container is the spin-resolved parallel that UMP2 (and later
/// UCCSD) needs.
///
/// Flat-index layout (C-order, the convention the [`ump2_kernel`] producer and
/// the RDM consumer both use):
/// - `t2aa` is `[nocc_a, nocc_a, nvir_a, nvir_a]`; element `(i,j,a,b)` lives at
///   `((i*nocc_a + j)*nvir_a + a)*nvir_a + b`.
/// - `t2ab` is `[nocc_a, nocc_b, nvir_a, nvir_b]`; element `(i,J,a,B)` lives at
///   `((i*nocc_b + J)*nvir_a + a)*nvir_b + B`.
/// - `t2bb` is `[nocc_b, nocc_b, nvir_b, nvir_b]`; element `(I,J,A,B)` lives at
///   `((I*nocc_b + J)*nvir_b + A)*nvir_b + B`.
#[derive(Debug, Clone, PartialEq)]
pub struct UmpAmplitudes {
    /// Number of (active) α-occupied orbitals.
    pub nocc_a: usize,
    /// Number of (active) α-virtual orbitals.
    pub nvir_a: usize,
    /// Number of (active) β-occupied orbitals.
    pub nocc_b: usize,
    /// Number of (active) β-virtual orbitals.
    pub nvir_b: usize,
    /// Same-spin αα amplitudes, flat `[nocc_a,nocc_a,nvir_a,nvir_a]` C-order.
    /// Antisymmetrized: `t2aa[i,j,a,b] = t2i[j,a,b] − t2i[j,b,a]` (`ump2.py:77`).
    pub t2aa: Vec<f64>,
    /// Opposite-spin αβ amplitudes, flat `[nocc_a,nocc_b,nvir_a,nvir_b]`.
    pub t2ab: Vec<f64>,
    /// Same-spin ββ amplitudes, flat `[nocc_b,nocc_b,nvir_b,nvir_b]` C-order.
    pub t2bb: Vec<f64>,
}

/// Unrestricted MP2 reference: the α/β reference pair (`ump2.py` carries
/// `(mo_a, mo_b)` via `mo_coeff[0/1]`, `mo_energy[0/1]`, `mo_occ[0/1]`).
///
/// We reuse [`Mp2Reference`] TWICE — once per spin channel — rather than a
/// flattened pair, so each channel carries its own coefficients/energies/occ
/// (matching upstream's `mo_*[0]` = α, `mo_*[1]` = β indexing). The `mol`
/// snapshot is shared content (identical molecule); only the spin-resolved MO
/// data differs between `alpha` and `beta`.
#[derive(Debug, Clone)]
pub struct UmpReference {
    /// α-spin reference snapshot.
    pub alpha: Mp2Reference,
    /// β-spin reference snapshot.
    pub beta: Mp2Reference,
}

/// Result of the open-shell UMP2 kernel (`ump2.py:107`'s
/// `tag_array(emp2_ss+emp2_os, e_corr_ss=..., e_corr_os=...)` decomposed by
/// spin channel).
#[derive(Debug, Clone)]
pub struct UmpResult {
    /// Total correlation energy `e_corr = e_aa + e_bb + e_ab`.
    pub e_corr: f64,
    /// Same-spin αα component.
    pub e_aa: f64,
    /// Opposite-spin αβ component.
    pub e_ab: f64,
    /// Same-spin ββ component.
    pub e_bb: f64,
    /// Spin-resolved amplitudes when `with_t2` was requested (`WITH_T2`).
    pub t2: Option<UmpAmplitudes>,
}

/// Active occupied / virtual MO energies for one spin channel, in column order.
///
/// Returns `(e_occ, e_vir)` — the active occupied energies followed by the
/// active virtual energies, honouring the frozen mask (mirrors the
/// `rmp2_kernel` energy split). `nocc`/`nvir` are returned implicitly via the
/// vector lengths.
fn channel_energies(
    refr: &Mp2Reference,
    frozen: &Frozen,
) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
    let elements: Vec<u32> = refr
        .mol
        .atom_charges()
        .into_iter()
        .map(|z| z.max(0) as u32)
        .collect();
    let mask = frozen::frozen_mask(frozen, &refr.mo_occ, &refr.mo_energy, &elements)?;
    let mut e_occ = Vec::new();
    let mut e_vir = Vec::new();
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
    Ok((e_occ, e_vir))
}

/// Validate that a [`ChemistsEris`] block matches the expected spin-block shape.
fn check_block(
    eris: &ChemistsEris,
    nocc_left: usize,
    nvir_left: usize,
    nocc_right: usize,
    nvir_right: usize,
    label: &str,
) -> Result<(), PyscfRsError> {
    // The block is the `(i,a,J,B)` ordering: left = (i,a), right = (J,B).
    // For the same-spin channels nocc_left==nocc_right and nvir_left==nvir_right.
    if eris.nocc != nocc_left {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: nocc_left,
            got: eris.nocc,
        }
        .into());
    }
    if eris.nvir != nvir_left {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: nvir_left,
            got: eris.nvir,
        }
        .into());
    }
    let expected_len = nocc_left * nvir_left * nocc_right * nvir_right;
    if eris.ovov.len() != expected_len {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: expected_len,
            got: eris.ovov.len(),
        }
        .into());
    }
    debug_assert!(!label.is_empty());
    Ok(())
}

/// Same-spin energy + (optionally) antisymmetrized amplitudes for one channel
/// (`ump2.py:64-78` for αα, `91-103` for ββ).
///
/// `eris.ovov` is the flat `(i,a,j,b)` C-order block (element `(i,a,j,b)` at
/// `((i*nvir + a)*nocc + j)*nvir + b`). For each occupied `i`:
/// `t2i[j,a,b] = (ia|jb) / (εi_a + εj_a − εa − εb)`,
/// `e += 0.5·Σ_jab t2i·(ia|jb) − 0.5·Σ_jab t2i·(ib|ja)`,
/// and `t2[i,j,a,b] = t2i[j,a,b] − t2i[j,b,a]` (antisymmetrized).
///
/// Returns `(e_same_spin, Option<t2_block>)` where the block is flat
/// `[nocc,nocc,nvir,nvir]` C-order.
fn same_spin_channel(
    eris: &ChemistsEris,
    e_occ: &[f64],
    e_vir: &[f64],
    with_t2: bool,
) -> (f64, Option<Vec<f64>>) {
    let nocc = eris.nocc;
    let nvir = eris.nvir;
    // Flat (i,a,j,b) index into ovov.
    let g =
        |i: usize, a: usize, j: usize, b: usize| eris.ovov[((i * nvir + a) * nocc + j) * nvir + b];
    let mut t2_store: Option<Vec<f64>> = if with_t2 {
        Some(vec![0.0; nocc * nocc * nvir * nvir])
    } else {
        None
    };
    // Per-i contributions, accumulated into a Vec then oracle_sum'd.
    let mut e_terms: Vec<f64> = Vec::with_capacity(nocc);

    let block = nocc * nvir * nvir; // jab span for a fixed i
    let mut gi: Vec<f64> = vec![0.0; block]; // gi[(j,a,b)] = (ia|jb)
    let mut g_jba: Vec<f64> = vec![0.0; block]; // (ib|ja) reordered to (j,a,b)
    let mut t2i: Vec<f64> = vec![0.0; block]; // t2i[(j,a,b)]

    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let p = (j * nvir + a) * nvir + b;
                    gi[p] = g(i, a, j, b);
                    // (ib|ja): swap the virtual roles a<->b.
                    g_jba[p] = g(i, b, j, a);
                    let denom = e_occ[i] + e_occ[j] - e_vir[a] - e_vir[b];
                    t2i[p] = gi[p] / denom;
                }
            }
        }
        // e += 0.5·Σ t2i·gi − 0.5·Σ t2i·(ib|ja).
        let e_direct = 0.5 * oracle_dot(&t2i, &gi);
        let e_exch = -0.5 * oracle_dot(&t2i, &g_jba);
        e_terms.push(e_direct + e_exch);

        if let Some(t2) = t2_store.as_mut() {
            // t2[i,j,a,b] = t2i[j,a,b] − t2i[j,b,a] (antisymmetrized, ump2.py:77).
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let p_ab = (j * nvir + a) * nvir + b;
                        let p_ba = (j * nvir + b) * nvir + a;
                        let dst = ((i * nocc + j) * nvir + a) * nvir + b;
                        t2[dst] = t2i[p_ab] - t2i[p_ba];
                    }
                }
            }
        }
    }
    (oracle_sum(&e_terms), t2_store)
}

/// Opposite-spin (αβ) energy + amplitudes (`ump2.py:79-89`).
///
/// `eris.ovOV` is the flat `(i,a,J,B)` C-order block: element `(i,a,J,B)` at
/// `((i*nvir_a + a)*nocc_b + J)*nvir_b + B`. For each α-occupied `i`:
/// `t2i[J,a,B] = (ia|JB) / (εi_a + εJ_b − εa_a − εB_b)`,
/// `e += Σ_JaB t2i·(ia|JB)` (DIRECT only, no exchange — opposite spins do not
/// antisymmetrize), and `t2ab[i,J,a,B] = t2i[J,a,B]`.
///
/// Returns `(e_os, Option<t2ab_block>)` flat `[nocc_a,nocc_b,nvir_a,nvir_b]`.
//
// The nested loops index four distinct energy slices AND compute a 4-D flat
// index in the same body; the raw indices are load-bearing for the flat-index
// layout (T-05-04-LAYOUT), so the `needless_range_loop` enumerate rewrite would
// obscure the index discipline rather than clarify it.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)]
fn opposite_spin_channel(
    eris: &ChemistsEris,
    nocc_a: usize,
    nvir_a: usize,
    nocc_b: usize,
    nvir_b: usize,
    e_occ_a: &[f64],
    e_vir_a: &[f64],
    e_occ_b: &[f64],
    e_vir_b: &[f64],
    with_t2: bool,
) -> (f64, Option<Vec<f64>>) {
    // Flat (i,a,J,B) index into the αβ ovOV block.
    let g = |i: usize, a: usize, jj: usize, bb: usize| {
        eris.ovov[((i * nvir_a + a) * nocc_b + jj) * nvir_b + bb]
    };
    let mut t2_store: Option<Vec<f64>> = if with_t2 {
        Some(vec![0.0; nocc_a * nocc_b * nvir_a * nvir_b])
    } else {
        None
    };
    let mut e_terms: Vec<f64> = Vec::with_capacity(nocc_a);

    let block = nocc_b * nvir_a * nvir_b; // (J,a,B) span for a fixed i
    let mut gi: Vec<f64> = vec![0.0; block]; // gi[(J,a,B)] = (ia|JB)
    let mut t2i: Vec<f64> = vec![0.0; block];

    for i in 0..nocc_a {
        for jj in 0..nocc_b {
            for a in 0..nvir_a {
                for bb in 0..nvir_b {
                    let p = (jj * nvir_a + a) * nvir_b + bb;
                    gi[p] = g(i, a, jj, bb);
                    let denom = e_occ_a[i] + e_occ_b[jj] - e_vir_a[a] - e_vir_b[bb];
                    t2i[p] = gi[p] / denom;
                }
            }
        }
        // e += Σ t2i·gi (direct only — opposite spin, no exchange).
        e_terms.push(oracle_dot(&t2i, &gi));

        if let Some(t2) = t2_store.as_mut() {
            // t2ab[i,J,a,B] = t2i[J,a,B] (ump2.py:89).
            for jj in 0..nocc_b {
                for a in 0..nvir_a {
                    for bb in 0..nvir_b {
                        let p = (jj * nvir_a + a) * nvir_b + bb;
                        let dst = ((i * nocc_b + jj) * nvir_a + a) * nvir_b + bb;
                        t2[dst] = t2i[p];
                    }
                }
            }
        }
    }
    (oracle_sum(&e_terms), t2_store)
}

/// Open-shell UMP2 correlation energy from the three spin-resolved `(ia|jb)`
/// blocks (`pyscf/mp/ump2.py:kernel`, 35-109).
///
/// `eris_aa` is the αα `(ia|jb)` block (`[nocc_a,nvir_a,nocc_a,nvir_a]`),
/// `eris_bb` the ββ block (`[nocc_b,nvir_b,nocc_b,nvir_b]`), and `eris_ab` the
/// αβ `(ia|JB)` block (`[nocc_a,nvir_a,nocc_b,nvir_b]`). The occupied/virtual
/// counts are taken from the active frozen mask of each channel; the supplied
/// blocks MUST match those counts (validated → `ShapeMismatch`, never OOB).
///
/// `e_corr = oracle_sum([e_aa, e_ab, e_bb])`. With `with_t2` the result carries
/// the [`UmpAmplitudes`] triple.
///
/// # Errors
/// Returns [`PyscfRsError`] (via [`crate::error::Mp2Error`]) on a spin-block
/// shape mismatch or a frozen-mask resolution failure. Never panics, never
/// indexes OOB (T-05-04-FFI).
pub fn ump2_kernel(
    refr: &UmpReference,
    frozen: &Frozen,
    eris_aa: &ChemistsEris,
    eris_ab: &ChemistsEris,
    eris_bb: &ChemistsEris,
    with_t2: bool,
) -> Result<UmpResult, PyscfRsError> {
    // Active occupied counts per spin via the MP2-08 helper (frozen-aware).
    let nocc_a = helpers::get_nocc(&refr.alpha.mo_occ, frozen)?;
    let nocc_b = helpers::get_nocc(&refr.beta.mo_occ, frozen)?;

    // Per-channel active orbital energies (frozen-aware).
    let (e_occ_a, e_vir_a) = channel_energies(&refr.alpha, frozen)?;
    let (e_occ_b, e_vir_b) = channel_energies(&refr.beta, frozen)?;
    let nvir_a = e_vir_a.len();
    let nvir_b = e_vir_b.len();

    // The energy-derived counts must agree with the helper-derived occ counts.
    if e_occ_a.len() != nocc_a {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: nocc_a,
            got: e_occ_a.len(),
        }
        .into());
    }
    if e_occ_b.len() != nocc_b {
        return Err(crate::error::Mp2Error::ShapeMismatch {
            expected: nocc_b,
            got: e_occ_b.len(),
        }
        .into());
    }

    // Validate spin-block shapes before any indexing.
    check_block(eris_aa, nocc_a, nvir_a, nocc_a, nvir_a, "aa")?;
    check_block(eris_bb, nocc_b, nvir_b, nocc_b, nvir_b, "bb")?;
    check_block(eris_ab, nocc_a, nvir_a, nocc_b, nvir_b, "ab")?;

    // Same-spin αα and ββ (antisymmetrized).
    let (e_aa, t2aa) = same_spin_channel(eris_aa, &e_occ_a, &e_vir_a, with_t2);
    let (e_bb, t2bb) = same_spin_channel(eris_bb, &e_occ_b, &e_vir_b, with_t2);

    // Opposite-spin αβ (direct only).
    let (e_ab, t2ab) = opposite_spin_channel(
        eris_ab, nocc_a, nvir_a, nocc_b, nvir_b, &e_occ_a, &e_vir_a, &e_occ_b, &e_vir_b, with_t2,
    );

    // e_corr = e_aa + e_ab + e_bb (deterministic order via oracle_sum).
    let e_corr = oracle_sum(&[e_aa, e_ab, e_bb]);

    let t2 = if with_t2 {
        Some(UmpAmplitudes {
            nocc_a,
            nvir_a,
            nocc_b,
            nvir_b,
            t2aa: t2aa.unwrap_or_default(),
            t2ab: t2ab.unwrap_or_default(),
            t2bb: t2bb.unwrap_or_default(),
        })
    } else {
        None
    };

    Ok(UmpResult {
        e_corr,
        e_aa,
        e_ab,
        e_bb,
        t2,
    })
}

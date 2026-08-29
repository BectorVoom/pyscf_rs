//! Even-tempered auxiliary basis generation — `pyscf/df/addons.py:76-228`.
//!
//! # Why this exists
//!
//! [`crate::auxbasis::DEFAULT_AUXBASIS`] only covers orbital bases that ship an
//! optimised fitting set. `gth-szv` — the basis every PBC reference system in
//! this workspace uses — has no such entry, so upstream falls back to an
//! **even-tempered** (ETB) auxiliary basis: for each `l`, `n` primitives with
//! exponents `emin · beta^k`. Phase 14's `GDF` cannot build an auxiliary cell
//! without it.
//!
//! # The two branches, and which one this port takes
//!
//! `_aug_etb_element` has a `USE_VERSION_26_AUXBASIS` switch. In the vendored
//! PySCF **2.12.1** that flag defaults to **`true`**, so the geometric-average
//! branch is the live one and the only one ported here. The arithmetic-average
//! branch (`false`) generates a different, larger auxiliary basis; a caller that
//! needs it gets [`PyscfRsError::NotYetImplemented`] rather than the wrong set.
//!
//! # Verified against upstream
//!
//! `C` / `gth-szv`, `beta = 2.0`:
//! `etb = [(0, 6, 0.2375755314, 2.0), (1, 6, …), (2, 6, …)]` → 36 shells /
//! 108 AOs for the two-atom diamond cell. Pinned in
//! `crates/pyscf-df/tests/etb_auxbasis.rs`.

use std::collections::HashMap;

use pyscf_core::{CoreError, ParsedBasis, PyscfRsError, ShellSpec};

/// `ETB_BETA` — `addons.py:33`. NOTE upstream swapped the two `getattr` keys
/// (`df_addons_aug_etb_beta` carries `DFBASIS` and vice versa); the VALUES are
/// what matter and they are `'weigend'` and `2.0`.
pub const ETB_BETA: f64 = 2.0;

/// `USE_VERSION_26_AUXBASIS` — `addons.py`. `true` in the vendored 2.12.1.
pub const USE_VERSION_26_AUXBASIS: bool = true;

/// The contraction-coefficient floor `_aug_etb_element` screens primitives at
/// (`addons.py`: `es = es[abs(cs).max(axis=1) > 1e-3]`).
pub const ETB_COEFF_CUTOFF: f64 = 1e-3;

use crate::configuration::CONFIGURATION;

/// `(l, n, emin, beta)` — one even-tempered block, as `_aug_etb_element` returns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EtbBlock {
    /// Angular momentum.
    pub l: u8,
    /// Number of primitives.
    pub n: usize,
    /// Smallest exponent; the block spans `emin · beta^k`, `k = 0..n`.
    pub emin: f64,
    /// Geometric ratio.
    pub beta: f64,
}

/// `_aug_etb_element(nuc_charge, basis, beta)` — `addons.py:76-135`.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] when `basis` is empty, and
/// [`PyscfRsError::NotYetImplemented`] for the `USE_VERSION_26_AUXBASIS = false`
/// branch (see the module docs).
pub fn aug_etb_element(
    nuc_charge: usize,
    basis: &ParsedBasis,
    beta: f64,
) -> Result<Vec<EtbBlock>, PyscfRsError> {
    if !USE_VERSION_26_AUXBASIS {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 14,
            what: "the arithmetic-average branch of _aug_etb_element \
                   (USE_VERSION_26_AUXBASIS = false)",
        });
    }
    if basis.shells.is_empty() {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "aug_etb_element: the orbital basis has no shells".into(),
        )));
    }

    let l_max_orb = basis.shells.iter().map(|s| s.l).max().unwrap_or(0) as usize;
    let mut emin_by_l = vec![f64::INFINITY; l_max_orb + 1];
    let mut emax_by_l = vec![0.0_f64; l_max_orb + 1];

    for sh in &basis.shells {
        let l = sh.l as usize;
        // `es = es[abs(cs).max(axis=1) > 1e-3]` — cs is [nprim][nctr] after the
        // transpose upstream's `e_c[:,1:]` implies, so the max runs over the
        // CONTRACTIONS of one primitive.
        for (p, &e) in sh.exponents.iter().enumerate() {
            let cmax = sh
                .coeffs
                .iter()
                .filter_map(|ctr| ctr.get(p))
                .fold(0.0_f64, |a, c| a.max(c.abs()));
            if cmax <= ETB_COEFF_CUTOFF {
                continue;
            }
            if e > emax_by_l[l] {
                emax_by_l[l] = e;
            }
            if e < emin_by_l[l] {
                emin_by_l[l] = e;
            }
        }
    }
    if emin_by_l.iter().any(|e| !e.is_finite()) {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "aug_etb_element: every primitive of some l <= {l_max_orb} was screened \
             out at |c| > {ETB_COEFF_CUTOFF}"
        ))));
    }

    let conf = CONFIGURATION.get(nuc_charge).ok_or_else(|| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "aug_etb_element: no CONFIGURATION row for Z = {nuc_charge}"
        )))
    })?;
    // `max_shells = 4 - conf.count(0)`.
    let max_shells = 4 - conf.iter().filter(|c| **c == 0).count();

    // The version-2.6 branch: geometric averages, and l_max clipped to max_shells.
    let l_max = l_max_orb.min(max_shells);
    let l_max_aux = l_max * 2;
    let l_max1 = l_max + 1;

    // `emax = sqrt(emax_i * emax_j) * 2`, `emin = sqrt(emin_i * emin_j) * 2`,
    // then reduced over `li + lj == ll`.
    let mut emax_ll = vec![f64::NEG_INFINITY; l_max_aux + 1];
    let mut emin_ll = vec![f64::INFINITY; l_max_aux + 1];
    for li in 0..l_max1 {
        for lj in 0..l_max1 {
            let ll = li + lj;
            let ex = (emax_by_l[li] * emax_by_l[lj]).sqrt() * 2.0;
            let en = (emin_by_l[li] * emin_by_l[lj]).sqrt() * 2.0;
            if ex > emax_ll[ll] {
                emax_ll[ll] = ex;
            }
            if en < emin_ll[ll] {
                emin_ll[ll] = en;
            }
        }
    }

    let mut etb = Vec::with_capacity(l_max_aux + 1);
    for ll in 0..=l_max_aux {
        // `ns = log((emax + emin) / emin) / log(beta)`, then `ceil`.
        let ns = ((emax_ll[ll] + emin_ll[ll]) / emin_ll[ll]).ln() / beta.ln();
        let n = ns.ceil();
        if n > 0.0 {
            etb.push(EtbBlock {
                l: u8::try_from(ll).map_err(|_| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "aug_etb_element: l = {ll} does not fit in u8"
                    )))
                })?,
                n: n as usize,
                emin: emin_ll[ll],
                beta,
            });
        }
    }
    Ok(etb)
}

/// `gto.expand_etbs(etbs)` — `pyscf/gto/mole.py`. One UNCONTRACTED shell per
/// primitive, **descending** in exponent (`emin · beta^(n-1)` first), which is
/// the AO ordering every downstream index assumes.
pub fn expand_etbs(etb: &[EtbBlock]) -> ParsedBasis {
    let mut shells = Vec::new();
    for b in etb {
        for k in (0..b.n).rev() {
            shells.push(ShellSpec {
                l: b.l,
                exponents: vec![b.emin * b.beta.powi(k as i32)],
                coeffs: vec![vec![1.0]],
            });
        }
    }
    ParsedBasis { shells }
}

/// `aug_etb(mol, beta)` — `addons.py:166-168`, i.e. `aug_etb_for_dfbasis` with
/// `start_at = 0` so EVERY element gets an even-tempered set.
///
/// Keys are the element symbols exactly as they appear in `_basis`.
///
/// # Errors
/// As [`aug_etb_element`], plus [`CoreError::InvalidMolecule`] when an atom's
/// element has no entry in `basis`.
pub fn aug_etb(
    atoms: &[pyscf_core::ParsedAtom],
    basis: &HashMap<String, ParsedBasis>,
    charge_of: impl Fn(&str) -> Option<usize>,
    beta: f64,
) -> Result<HashMap<String, ParsedBasis>, PyscfRsError> {
    let mut out: HashMap<String, ParsedBasis> = HashMap::new();
    for (sym, _) in atoms {
        if out.contains_key(sym) {
            continue;
        }
        // `_basis` keys are the element symbol as the basis loader normalised
        // it, which is UPPER-CASE for two-letter elements ("HE") while `_atom`
        // keeps the input casing ("He"). Match upstream's `_rm_digit` fallback
        // by trying the exact key first and then case-insensitively.
        let obs = basis
            .get(sym)
            .or_else(|| {
                basis
                    .iter()
                    .find(|(k, _)| k.eq_ignore_ascii_case(sym))
                    .map(|(_, v)| v)
            })
            .ok_or_else(|| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "aug_etb: element '{sym}' has no entry in _basis (keys: {:?})",
                    basis.keys().collect::<Vec<_>>()
                )))
            })?;
        let z = charge_of(sym).ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "aug_etb: unknown element symbol '{sym}'"
            )))
        })?;
        let etb = aug_etb_element(z, obs, beta)?;
        if etb.is_empty() {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "aug_etb: failed to generate an even-tempered auxbasis for '{sym}'"
            ))));
        }
        out.insert(sym.clone(), expand_etbs(&etb));
    }
    Ok(out)
}

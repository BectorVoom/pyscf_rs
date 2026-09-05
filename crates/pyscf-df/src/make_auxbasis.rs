//! `df.addons.make_auxbasis` / `predefined_auxbasis` / `bse_predefined_auxbasis`
//! — `pyscf/df/addons.py:170-228` (plan 14-01 Task 3).
//!
//! # The resolution chain, in upstream's order
//!
//! 1. **Psi4 recommendations** — [`crate::psi4_auxbasis::PSI4_AUXBASIS`],
//!    keyed by the CANONICAL orbital-basis name.
//! 2. **Basis Set Exchange metadata** — [`crate::bse_auxbasis::BSE_AUXBASIS`].
//! 3. **Even-tempered fallback** — [`crate::etb::aug_etb`], per element, for
//!    every element the first two steps could not place.
//!
//! Upstream applies steps 1–2 per element and only falls through to step 3 if
//! *some* element is still unplaced, in which case it generates ETB for the
//! WHOLE molecule and then re-overrides the elements it did place
//! (`addons.py:217-227`). That order matters and is reproduced here.
//!
//! # Measured on the two Phase-14 reference cells
//!
//! | cell | orbital basis | route | result |
//! |---|---|---|---|
//! | diamond | `gth-szv` | ETB (no table entry) | 18 shells / 54 AOs per C |
//! | He-fcc | `sto-3g` | Psi4 → `def2-svp-jkfit` | 9 shells / 23 AOs per He |
//!
//! Both are pinned in `crates/pyscf-df/tests/make_auxbasis.rs`.

use std::collections::HashMap;

use pyscf_core::{CoreError, ParsedAtom, ParsedBasis, PyscfRsError};

use crate::bse_auxbasis::BSE_AUXBASIS;
use crate::etb::{ETB_BETA, aug_etb};
use crate::psi4_auxbasis::PSI4_AUXBASIS;

/// `predefined_auxbasis(mol, basis, xc, mp2fit)` — `addons.py:170-214` plus its
/// `bse_predefined_auxbasis` tail.
///
/// `hybrid_xc` is upstream's `is_hybrid_xc(xc)`; the default `xc = 'HF'` makes
/// it `true`, which is the branch every DF-for-JK caller takes.
pub fn predefined_auxbasis(basis: &str, hybrid_xc: bool, mp2fit: bool) -> Option<&'static str> {
    let key = pyscf_gto::basis::canonicalise_basis_name(basis);
    if let Some((_, jkfit, rifit)) = PSI4_AUXBASIS.iter().find(|(k, _, _)| *k == key) {
        if mp2fit {
            return Some(rifit);
        } else if hybrid_xc {
            return Some(jkfit);
        }
    }
    bse_predefined_auxbasis(&key, hybrid_xc, mp2fit)
}

/// `bse_predefined_auxbasis` — `addons.py:145-168`. `key` must already be
/// canonical.
fn bse_predefined_auxbasis(key: &str, hybrid_xc: bool, mp2fit: bool) -> Option<&'static str> {
    let (_, jkfit, jfit, dftjfit, rifit) = BSE_AUXBASIS.iter().find(|(k, ..)| *k == key)?;
    if mp2fit {
        *rifit
    } else if hybrid_xc {
        *jkfit
    } else {
        // `jfit` → `dftjfit` → `jkfit`, in that order (addons.py:161-167).
        jfit.or(*dftjfit).or(*jkfit)
    }
}

/// `make_auxbasis(mol, xc='HF', mp2fit=False)` — `addons.py:170-227`, resolved
/// all the way to parsed shells.
///
/// * `atoms` — `mol._atom`, for the element list.
/// * `basis_names` — the orbital-basis NAME per element symbol. An element
///   whose basis was given as raw text or as parsed shells has no name and is
///   sent straight to the even-tempered fallback, which is what upstream's
///   `if not isinstance(obs, str): continue` does.
/// * `parsed_basis` — `mol._basis`, needed by the ETB fallback.
/// * `charge_of` — element symbol → nuclear charge.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] when an element is missing from
/// `parsed_basis`, and anything [`aug_etb`] can raise.
pub fn make_auxbasis(
    atoms: &[ParsedAtom],
    basis_names: &HashMap<String, String>,
    parsed_basis: &HashMap<String, ParsedBasis>,
    charge_of: impl Fn(&str) -> Option<usize> + Copy,
    hybrid_xc: bool,
    mp2fit: bool,
) -> Result<HashMap<String, ParsedBasis>, PyscfRsError> {
    let mut uniq: Vec<String> = Vec::new();
    for (sym, _) in atoms {
        if !uniq.iter().any(|s| s == sym) {
            uniq.push(sym.clone());
        }
    }

    // Steps 1-2: the tables, per element. An entry only counts as PLACED if the
    // named auxiliary basis actually loads for that element — upstream wraps the
    // lookup in `try: gto.basis.load(...) except BasisNotFoundError` and drops
    // it otherwise (addons.py:196-214).
    let mut placed: HashMap<String, ParsedBasis> = HashMap::new();
    for sym in &uniq {
        let Some(name) = basis_names.get(sym) else {
            continue;
        };
        let Some(auxb) = predefined_auxbasis(name, hybrid_xc, mp2fit) else {
            continue;
        };
        // `_local`: this is a probe, and an unanswered probe must fall to ETB
        // rather than reach for the network.
        match pyscf_gto::basis::load_basis_local(auxb, sym) {
            Ok(b) if !b.shells.is_empty() => {
                tracing::debug!(
                    target: "pyscf_df::make_auxbasis",
                    element = sym.as_str(), orbital = name.as_str(), aux = auxb,
                    "predefined auxbasis"
                );
                placed.insert(sym.clone(), b);
            }
            _ => {
                tracing::debug!(
                    target: "pyscf_df::make_auxbasis",
                    element = sym.as_str(), aux = auxb,
                    "predefined auxbasis does not cover this element; falling back to ETB"
                );
            }
        }
    }

    if placed.len() == uniq.len() {
        return Ok(placed);
    }

    // Step 3: `auxbasis, auxdefault = aug_etb(mol), auxbasis; auxbasis.update(
    // auxdefault)` — ETB for EVERY element, then the placed ones win.
    let mut out = aug_etb(atoms, parsed_basis, charge_of, ETB_BETA)?;
    let etb_only: Vec<&String> = uniq.iter().filter(|s| !placed.contains_key(*s)).collect();
    if !etb_only.is_empty() {
        tracing::warn!(
            target: "pyscf_df::make_auxbasis",
            "even-tempered Gaussians generated as DF auxbasis for {:?}", etb_only
        );
    }
    for (k, v) in placed {
        out.insert(k, v);
    }
    if out.len() != uniq.len() {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "make_auxbasis: resolved {} of {} elements",
            out.len(),
            uniq.len()
        ))));
    }
    Ok(out)
}

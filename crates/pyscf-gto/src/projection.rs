//! GTO-11: Build `cintx_core::BasisSet` (Arc-backed) from `mol._basis` + `mol._atom`.
//!
//! After construction, `Arc<BasisSet>` is stored in `mol.basis_set` for
//! zero-copy access by 02-05 intor + 02-06 eval_gto + downstream phase
//! consumers (SCF, DFT, MP2, CCSD, grad).
//!
//! The cintx `BasisSet` is the SINGLE source of truth for the typed basis
//! per D-03 + GTO-11. pyscf-rs does NOT maintain a parallel basis structure.
//!
//! Coefficients are normalised through `make_env::normalise_contractions`
//! before being handed to cintx — this matches the `_env`-side normalisation
//! so all downstream consumers (libcint via cintx-compat AND the typed cintx
//! BasisSet) see the same coefficient values.

use cintx_core::{
    Atom as CintxAtom, BasisSet, NuclearModel as CintxNuc, Representation, Shell as CintxShell,
};
use pyscf_core::{CoreError, ParsedAtom, ParsedBasis, PyscfRsError};
use std::collections::HashMap;
use std::sync::Arc;

/// Build the `Arc<BasisSet>` projection of a `(atoms, basis_per_symbol)` pair.
///
/// Per-symbol shells are emitted in first-occurrence order (Pitfall 4) and
/// cloned per atom — matching the layout produced by `make_env`.
pub fn build_cintx_basis_set(
    atoms: &[ParsedAtom],
    basis: &HashMap<String, ParsedBasis>,
    cart: bool,
) -> Result<Arc<BasisSet>, PyscfRsError> {
    let representation = if cart {
        Representation::Cart
    } else {
        Representation::Spheric
    };

    // Build cintx Atoms (typed, validated).
    let mut cintx_atoms: Vec<CintxAtom> = Vec::with_capacity(atoms.len());
    for (sym, xyz) in atoms {
        let alpha: String = sym.chars().take_while(|c| c.is_alphabetic()).collect();
        let raw_charge = crate::format_atom::charge_for_symbol(&alpha).unwrap_or(0);
        // cintx_core::Atom::try_new rejects atomic_number == 0. Ghost atoms keep
        // basis functions but have no nuclear charge; map them to atomic_number=1
        // for the cintx-core typed view (the libcint flat-array side already
        // carries CHARGE_OF=0 via _atm). Phase 2 test corpus has no ghost atoms.
        let atomic_number: u16 = if raw_charge <= 0 {
            1
        } else {
            raw_charge as u16
        };
        let atom =
            CintxAtom::try_new(atomic_number, *xyz, CintxNuc::Point, None, None).map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "cintx Atom::try_new failed for symbol '{sym}' (Z={atomic_number}): {e}"
                )))
            })?;
        cintx_atoms.push(atom);
    }

    // Build cintx Shells in per-symbol-first-occurrence order, cloned per atom.
    let mut cintx_shells: Vec<Arc<CintxShell>> = Vec::new();
    for (atom_id, (sym, _)) in atoms.iter().enumerate() {
        let alpha: String = sym.chars().take_while(|c| c.is_alphabetic()).collect();
        let upper = alpha.to_ascii_uppercase();
        let parsed = basis.get(&upper).ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "no parsed basis entry for symbol '{upper}'"
            )))
        })?;

        for spec in &parsed.shells {
            let l = spec.l;
            let nprim = spec.exponents.len();
            let nctr = spec.coeffs.len();
            if nprim == 0 || nctr == 0 {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "empty shell for symbol '{upper}': nprim={nprim} nctr={nctr}"
                ))));
            }

            // Apply the same gto_norm + _nomalize_contracted_ao normalisation
            // that make_env applies; cintx and the libcint flat-array view stay
            // in lockstep.
            let final_coeffs =
                crate::make_env::normalise_contractions(l, &spec.exponents, &spec.coeffs);

            // Flatten coeffs in F-order: outer = contraction column, inner = primitive.
            let mut coeffs_flat: Vec<f64> = Vec::with_capacity(nprim * nctr);
            for col in &final_coeffs {
                coeffs_flat.extend_from_slice(col);
            }

            let exps_arc: Arc<[f64]> = Arc::from(spec.exponents.clone().into_boxed_slice());
            let coeffs_arc: Arc<[f64]> = Arc::from(coeffs_flat.into_boxed_slice());

            let shell = CintxShell::try_new(
                atom_id as u32,
                l,
                nprim as u16,
                nctr as u16,
                0, // Pitfall 5: kappa=0 for sph/cart in v1
                representation,
                exps_arc,
                coeffs_arc,
            )
            .map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "cintx Shell::try_new failed for atom {atom_id} symbol '{upper}' \
                     (l={l} nprim={nprim} nctr={nctr}): {e}"
                )))
            })?;
            cintx_shells.push(Arc::new(shell));
        }
    }

    let atoms_arc: Arc<[CintxAtom]> = Arc::from(cintx_atoms.into_boxed_slice());
    let shells_arc: Arc<[Arc<CintxShell>]> = Arc::from(cintx_shells.into_boxed_slice());

    let basis_set = BasisSet::try_new(atoms_arc, shells_arc).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx BasisSet::try_new failed: {e}"
        )))
    })?;

    Ok(Arc::new(basis_set))
}

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

use cintx_core::ecp::{ECP_LMAX, EcpChannel, EcpShell as CintxEcpShell};
use cintx_core::{
    Atom as CintxAtom, BasisSet, NuclearModel as CintxNuc, Representation, Shell as CintxShell,
};
use pyscf_core::{CoreError, ParsedAtom, ParsedBasis, ParsedEcp, PyscfRsError};
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
    let (cintx_atoms, cintx_shells) = build_atoms_and_shells(atoms, basis, cart)?;

    let atoms_arc: Arc<[CintxAtom]> = Arc::from(cintx_atoms.into_boxed_slice());
    let shells_arc: Arc<[Arc<CintxShell>]> = Arc::from(cintx_shells.into_boxed_slice());

    let basis_set = BasisSet::try_new(atoms_arc, shells_arc).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx BasisSet::try_new failed: {e}"
        )))
    })?;

    Ok(Arc::new(basis_set))
}

/// GTO-05 (eval half, plan 02-10): build an ECP-augmented `Arc<BasisSet>`.
///
/// Identical AO-shell + atom construction to [`build_cintx_basis_set`], but
/// additionally projects `mol._ecp` (per-element `ParsedEcp`) into typed
/// `cintx_core::ecp::EcpShell`s and attaches them via
/// `BasisSet::try_new_with_ecp`. The cintx safe-API ECP preflight
/// (`SessionRequest::query_workspace`) requires `basis.ecp_shells()` to be
/// non-empty for the four `int1e_ecp_*` operators (else
/// `FacadeError::MissingEcpBasis`); the non-ECP `build_cintx_basis_set`
/// stored in `mol.basis_set` carries an empty ECP slab, so the
/// `CintxEcpEngine` builds this augmented view on demand.
///
/// ECP-shell projection mirrors `make_ecp_env`'s `_ecpbas` row grouping:
/// one cintx `EcpShell` per `(atom, channel, distinct n_power)` block, so
/// each carries clean per-radial-power exponent/coefficient slabs. The
/// pyscf-core `EcpShell.l` sentinel (`-1` = local/UL channel; `0..=ECP_LMAX`
/// = projector) maps onto `EcpChannel::Local` / `EcpChannel::Projected(l)`.
pub fn build_cintx_basis_set_with_ecp(
    atoms: &[ParsedAtom],
    basis: &HashMap<String, ParsedBasis>,
    ecp: &HashMap<String, ParsedEcp>,
    cart: bool,
) -> Result<Arc<BasisSet>, PyscfRsError> {
    let (cintx_atoms, cintx_shells) = build_atoms_and_shells(atoms, basis, cart)?;

    // Project per-element ParsedEcp into typed cintx EcpShells, keyed by the
    // same first-alpha-uppercase element-symbol convention make_ecp_env uses.
    let mut ecp_shells: Vec<Arc<CintxEcpShell>> = Vec::new();
    for (atom_id, (sym, _)) in atoms.iter().enumerate() {
        let alpha: String = sym.chars().take_while(|c| c.is_alphabetic()).collect();
        let upper = alpha.to_ascii_uppercase();
        let parsed = match ecp.get(&upper) {
            Some(p) => p,
            None => continue,
        };

        for channel in &parsed.channels {
            // Map the pyscf-core l sentinel onto the cintx EcpChannel enum.
            let cintx_channel = if channel.l < 0 {
                EcpChannel::Local
            } else {
                let l = channel.l as u8;
                if l > ECP_LMAX {
                    return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "ECP projector l={l} for symbol '{upper}' exceeds ECP_LMAX={ECP_LMAX}"
                    ))));
                }
                EcpChannel::Projected(l)
            };

            // Group primitives by distinct n_power, first-occurrence order —
            // exactly as make_ecp_env emits one _ecpbas row per group.
            let mut seen_orders: Vec<i32> = Vec::new();
            for &np in &channel.n_powers {
                if !seen_orders.contains(&np) {
                    seen_orders.push(np);
                }
            }
            for &order in &seen_orders {
                let mut exps_buf: Vec<f64> = Vec::new();
                let mut coefs_buf: Vec<f64> = Vec::new();
                for ((np, exp), coef) in channel
                    .n_powers
                    .iter()
                    .zip(channel.exponents.iter())
                    .zip(channel.coeffs.iter())
                {
                    if *np == order {
                        exps_buf.push(*exp);
                        coefs_buf.push(*coef);
                    }
                }
                if exps_buf.is_empty() {
                    continue;
                }

                let nprim = exps_buf.len() as u16;
                let exps_arc: Arc<[f64]> = Arc::from(exps_buf.into_boxed_slice());
                let coeffs_arc: Arc<[f64]> = Arc::from(coefs_buf.into_boxed_slice());

                let shell = CintxEcpShell::try_new(
                    atom_id as u32,
                    cintx_channel,
                    order as i16,
                    nprim,
                    1, // nctr — one contraction per radial-power block
                    0, // so_type — scalar (spin-orbit out of scope per Phase 19 D-12)
                    exps_arc,
                    coeffs_arc,
                )
                .map_err(|e| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "cintx EcpShell::try_new failed for atom {atom_id} symbol '{upper}' \
                         (l={} n_power={order}): {e}",
                        channel.l
                    )))
                })?;
                ecp_shells.push(Arc::new(shell));
            }
        }
    }

    let atoms_arc: Arc<[CintxAtom]> = Arc::from(cintx_atoms.into_boxed_slice());
    let shells_arc: Arc<[Arc<CintxShell>]> = Arc::from(cintx_shells.into_boxed_slice());
    let ecp_arc: Arc<[Arc<CintxEcpShell>]> = Arc::from(ecp_shells.into_boxed_slice());

    let basis_set = BasisSet::try_new_with_ecp(atoms_arc, shells_arc, ecp_arc).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx BasisSet::try_new_with_ecp failed: {e}"
        )))
    })?;

    Ok(Arc::new(basis_set))
}

/// Shared atom + AO-shell construction for the ECP and non-ECP basis-set
/// builders. Extracted by plan 02-10 so the ECP-augmented variant reuses
/// the exact same typed AO projection (Pitfall 4 first-occurrence order,
/// make_env-matching contraction normalisation).
fn build_atoms_and_shells(
    atoms: &[ParsedAtom],
    basis: &HashMap<String, ParsedBasis>,
    cart: bool,
) -> Result<(Vec<CintxAtom>, Vec<Arc<CintxShell>>), PyscfRsError> {
    build_atoms_and_shells_with_base(atoms, basis, representation_of(cart), 0)
}

/// Map the `cart` flag to the scalar cintx representation. Spinor shells are
/// built via [`build_cintx_spinor_basis_set`] (which passes
/// `Representation::Spinor` directly), not through this helper.
fn representation_of(cart: bool) -> Representation {
    if cart {
        Representation::Cart
    } else {
        Representation::Spheric
    }
}

/// F-03 (route-through-cintx): build a **Spinor**-tagged `Arc<BasisSet>` for
/// `intor_spinor`. Identical AO-shell + atom construction to
/// [`build_cintx_basis_set`] (same spherical contraction normalisation), but
/// each shell is tagged `Representation::Spinor` with `kappa = 0` (both
/// `j = l ± 1/2` blocks). cintx's `Shell::ao_per_shell` then reports the
/// 2-component `spinor_len = 2·(2l+1)·nctr`, so the resulting `BasisMeta`
/// offsets/counts are in `n2c` units and the cintx kernel applies its
/// libcint-parity-validated cart→spinor transform (vendor-gated oracle:
/// `cintx-oracle/tests/one_electron_scalar_spinor_parity.rs`).
///
/// Spinors are inherently spherical-harmonic based, so this path does not
/// consult `mol.cart` — the cart→spinor transform consumes cartesian gaussians
/// directly regardless.
pub fn build_cintx_spinor_basis_set(
    atoms: &[ParsedAtom],
    basis: &HashMap<String, ParsedBasis>,
) -> Result<Arc<BasisSet>, PyscfRsError> {
    let (cintx_atoms, cintx_shells) =
        build_atoms_and_shells_with_base(atoms, basis, Representation::Spinor, 0)?;

    let atoms_arc: Arc<[CintxAtom]> = Arc::from(cintx_atoms.into_boxed_slice());
    let shells_arc: Arc<[Arc<CintxShell>]> = Arc::from(cintx_shells.into_boxed_slice());

    let basis_set = BasisSet::try_new(atoms_arc, shells_arc).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx spinor BasisSet::try_new failed: {e}"
        )))
    })?;

    Ok(Arc::new(basis_set))
}

/// Combined two-basis `BasisSet` (the PySCF `conc_mol`/fakemol pattern,
/// `pyscf/df/incore.py` + `gto.mole.intor_cross`): the "A" shells occupy the
/// leading shell range and the "B" shells follow, with B shells re-based onto
/// the appended B atoms so the combined `BasisSet` owns one contiguous atom +
use pyscf_core::Mole;

/// Build a combined `BasisSet` from two `Mole` instances.
///
/// Shells of `mol_a` lead, shells of `mol_b` follow, with atom indices in B
/// shells offset by `mol_a.natm`.
///
/// Returns `(basis, n_a_shells, n_b_shells)`.
pub fn build_combined_basis(
    mol_a: &Mole,
    mol_b: &Mole,
) -> Result<(Arc<BasisSet>, usize, usize), PyscfRsError> {
    build_combined_basis_raw(
        &mol_a._atom,
        &mol_a._basis,
        mol_a.cart,
        &mol_b._atom,
        &mol_b._basis,
        mol_b.cart,
    )
}

/// Raw form taking atom and basis slices/maps directly.
///
/// Returns `(basis, n_a_shells, n_b_shells)`. The caller iterates the A range
/// `0..n_a_shells` and the B range `n_a_shells..`; A AOs occupy combined AO
/// range `[0, nao_a)` and B AOs `[nao_a, nao_a+nao_b)`, so a B shell's AO offset
/// within the B block is `shell_offset(j) - nao_a`. Used by int3c2e (orbital×aux,
/// the third center) and by `intor_cross` (cross-basis arity-2, e.g. overlap).
pub fn build_combined_basis_raw(
    a_atoms: &[ParsedAtom],
    a_basis: &HashMap<String, ParsedBasis>,
    a_cart: bool,
    b_atoms: &[ParsedAtom],
    b_basis: &HashMap<String, ParsedBasis>,
    b_cart: bool,
) -> Result<(Arc<BasisSet>, usize, usize), PyscfRsError> {
    let (mut atoms, a_shells) =
        build_atoms_and_shells_with_base(a_atoms, a_basis, representation_of(a_cart), 0)?;
    let n_a_atoms = atoms.len() as u32;
    let (b_atoms_cintx, b_shells) =
        build_atoms_and_shells_with_base(b_atoms, b_basis, representation_of(b_cart), n_a_atoms)?;

    atoms.extend(b_atoms_cintx);
    let n_a_shells = a_shells.len();
    let n_b_shells = b_shells.len();
    let mut shells = a_shells;
    shells.extend(b_shells);

    let atoms_arc: Arc<[CintxAtom]> = Arc::from(atoms.into_boxed_slice());
    let shells_arc: Arc<[Arc<CintxShell>]> = Arc::from(shells.into_boxed_slice());
    let basis_set = BasisSet::try_new(atoms_arc, shells_arc).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx combined two-basis BasisSet::try_new failed: {e}"
        )))
    })?;

    Ok((Arc::new(basis_set), n_a_shells, n_b_shells))
}

/// Build typed cintx atoms + shells, with each shell's `atom_index` offset by
/// `atom_id_base`. `atom_id_base == 0` reproduces the standalone-Mole layout;
/// a non-zero base lets the auxiliary shells reference atoms appended after
/// the orbital atoms in a combined `BasisSet` (see [`build_combined_basis`]).
fn build_atoms_and_shells_with_base(
    atoms: &[ParsedAtom],
    basis: &HashMap<String, ParsedBasis>,
    representation: Representation,
    atom_id_base: u32,
) -> Result<(Vec<CintxAtom>, Vec<Arc<CintxShell>>), PyscfRsError> {
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
            // in lockstep. `final_coeffs[ctr][prim]` is contraction-major.
            let final_coeffs =
                crate::make_env::normalise_contractions(l, &spec.exponents, &spec.coeffs);

            // Flatten coeffs into the layout the cintx 1e/2e kernel CONSUMES:
            // `coefficients[prim * nctr + ctr]` (primitive-major / contraction-
            // minor, i.e. ROW-MAJOR `[nprim][nctr]`). This is the cintx safe-API
            // Shell convention — see cintx `kernels/one_electron.rs`
            // (`shell.coefficients[pi * n_ctr + ci]`). For the SEGMENTED case
            // (nctr == 1) this interleave is byte-identical to a plain column
            // copy, which is why the truncating-parser era never exposed it; for
            // a GENERAL CONTRACTION (nctr > 1, e.g. ANO/cc-pVDZ O) a column-major
            // flatten silently SCRAMBLES the coefficients and yields a wrong,
            // un-normalised, asymmetric overlap (threat T-02-11-CINTX-NCTR).
            let mut coeffs_flat: Vec<f64> = vec![0.0; nprim * nctr];
            for (ctr, col) in final_coeffs.iter().enumerate() {
                for (prim, &c) in col.iter().enumerate() {
                    coeffs_flat[prim * nctr + ctr] = c;
                }
            }

            let exps_arc: Arc<[f64]> = Arc::from(spec.exponents.clone().into_boxed_slice());
            let coeffs_arc: Arc<[f64]> = Arc::from(coeffs_flat.into_boxed_slice());

            let shell = CintxShell::try_new(
                atom_id_base + atom_id as u32,
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

    Ok((cintx_atoms, cintx_shells))
}

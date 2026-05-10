//! GTO-09: `dumps()` / `loads()` — semantic JSON round-trip via `serde_json`.
//!
//! Per CONTEXT D-09 + RESEARCH "Don't Hand-Roll": the contract is
//! **semantic round-trip only** (read your own JSON, get the same internal
//! arrays). It is **NOT** byte-identical to upstream PySCF's JSON string —
//! Phase 8 ORACLE-08 chkfile interop covers cross-language compatibility.
//!
//! Strategy: serialise a `MoleSnapshot` containing the user-provided inputs
//! (`_atom` parsed coords, `_basis` parsed shells, `_ecp` parsed channels,
//! plus all scalar kwargs). On `loads`, re-run `pyscf_gto::M(args)` to
//! reconstruct every derived array (`_atm`/`_bas`/`_env`/`ao_loc_nr`/
//! `nao_nr`/`_ecpbas`) **byte-for-byte**, because `make_env` /
//! `format_basis` / `format_ecp` / `make_ecp_env` are pure deterministic
//! functions of those inputs.
//!
//! Source: `pyscf/gto/mole.py:1251-1349` (`dumps` / `loads`).

use crate::types::{AtomInput, BasisInput, EcpInput, MoleBuildArgs};
use pyscf_core::{CoreError, Mole, ParsedBasis, ParsedEcp, PyscfRsError, Unit};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Snapshot of the user-input portion of `Mole` state, suitable for
/// `serde_json` serialisation. Re-running `M(args)` on load reproduces
/// every derived array bit-for-bit.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MoleSnapshot {
    /// Parsed `(symbol, [x, y, z]_in_Bohr)` list — already canonicalised.
    /// Stored in Bohr so we can re-build with `unit = Bohr` and skip the
    /// unit conversion (keeps the round-trip a pure passthrough).
    atom_parsed: Vec<(String, [f64; 3])>,
    /// Per-element parsed basis (uppercase-keyed), so we round-trip the
    /// shells independent of the original basis-input form (Name vs.
    /// NwchemText vs. Parsed all collapse to this).
    basis_per_element: HashMap<String, ParsedBasis>,
    /// Per-element parsed ECP (uppercase-keyed). Empty if no ECP.
    ecp_per_element: HashMap<String, ParsedEcp>,
    /// Scalar kwargs.
    charge: i32,
    spin: i32,
    cart: bool,
    /// Original user `unit`. We store coords in Bohr regardless, but echo
    /// the original unit so `mol.unit` round-trips.
    unit: Unit,
    verbose: u8,
    max_memory: f64,
    output: Option<PathBuf>,
    /// Symmetry / groupname — out of v1 scope but echoed for forward-compat.
    symmetry: bool,
    groupname: String,
    topgroup: String,
}

/// Serialise a `Mole` to a JSON string.
///
/// Re-running `loads()` on the result reconstructs a `Mole` whose
/// `_atm` / `_bas` / `_env` / `ao_loc_nr` / `nao_nr` / `_ecpbas` equal the
/// original byte-for-byte.
///
/// Source: `pyscf/gto/mole.py:1251-1290` (`dumps`).
pub fn dumps(mol: &Mole) -> Result<String, PyscfRsError> {
    let snap = MoleSnapshot {
        atom_parsed: mol._atom.clone(),
        basis_per_element: mol._basis.clone(),
        ecp_per_element: mol._ecp.clone(),
        charge: mol.charge,
        spin: mol.spin,
        cart: mol.cart,
        unit: mol.unit,
        verbose: mol.verbose,
        max_memory: mol.max_memory,
        output: mol.output.clone(),
        symmetry: mol.symmetry,
        groupname: mol.groupname.clone(),
        topgroup: mol.topgroup.clone(),
    };
    serde_json::to_string(&snap).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "dumps serde_json error: {}",
            e
        )))
    })
}

/// Reconstruct a `Mole` from a JSON string produced by [`dumps`].
///
/// Source: `pyscf/gto/mole.py:1292-1349` (`loads`).
pub fn loads(json: &str) -> Result<Mole, PyscfRsError> {
    let snap: MoleSnapshot = serde_json::from_str(json).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "loads serde_json error: {}",
            e
        )))
    })?;

    // Re-construct `BasisInput::PerElement<Parsed>` so `format_basis` keys
    // line up with the saved `_basis` map (uppercase symbols). The resolver
    // (`format_basis::resolve_for_symbol`) does an exact-key lookup first,
    // matching our uppercase entries directly.
    let basis = if snap.basis_per_element.is_empty() {
        BasisInput::Name(String::new())
    } else {
        let mut per_elem: HashMap<String, BasisInput> =
            HashMap::with_capacity(snap.basis_per_element.len());
        for (sym, parsed) in snap.basis_per_element {
            per_elem.insert(sym, BasisInput::Parsed(parsed));
        }
        BasisInput::PerElement(per_elem)
    };

    // Same shape for ECP — empty map ⇒ EcpInput::None (don't trigger ECP path).
    let ecp = if snap.ecp_per_element.is_empty() {
        EcpInput::None
    } else {
        let mut per_elem: HashMap<String, EcpInput> =
            HashMap::with_capacity(snap.ecp_per_element.len());
        for (sym, parsed) in snap.ecp_per_element {
            per_elem.insert(sym, EcpInput::Parsed(parsed));
        }
        EcpInput::PerElement(per_elem)
    };

    let args = MoleBuildArgs {
        atom: AtomInput::Tuples(snap.atom_parsed),
        basis,
        ecp,
        charge: snap.charge,
        spin: snap.spin,
        cart: snap.cart,
        // Coords were stored in Bohr; force Bohr on rebuild so the
        // identity-axes/origin pass through `format_atom` is a no-op.
        unit: Unit::Bohr,
        verbose: snap.verbose,
        max_memory: snap.max_memory,
        output: snap.output,
        origin: [0.0; 3],
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    let mut mol = crate::M(args)?;
    // Restore the original user-facing unit (atomic state is in Bohr either
    // way; this is a label, not a conversion factor at this point).
    mol.unit = snap.unit;
    Ok(mol)
}

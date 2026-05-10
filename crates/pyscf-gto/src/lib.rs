//! pyscf-gto — Molecular structure & integrals.
//!
//! Phase 2 fills the bodies for GTO-01..11. Wave 0 (plan 02-01) laid out
//! modules and proved cintx reachability. Plan 02-02 (this commit) ships
//! the GTO-01 atom-input front-door + the GTO-08 ≥30-attribute Mole
//! floor wired via `pyscf_gto::M(MoleBuildArgs { ... })`.

#![forbid(unsafe_code)]

pub mod format_atom;
pub mod layout_table; // Wave 0 (plan 02-01); consumed by intor.rs in 02-05.
pub mod types;

// Plans 02-03..02-08 add: format_basis, basis (mod), make_env, intor,
// eval_gto, ecp_engine_stub, dumps_loads.

pub use pyscf_core::{Mole, Unit};
pub use types::{AtomInput, BasisInput, EcpInput, MoleBuildArgs};

/// Shortcut to build a Mole. Equivalent to `pyscf.M(...)` upstream.
///
/// Source: `pyscf/gto/mole.py:106-118` (Apache-2.0).
///
/// # Example
///
/// ```ignore
/// use pyscf_core::Unit;
/// use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};
///
/// let mol = M(MoleBuildArgs {
///     atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
///     basis: BasisInput::Name("sto-3g".into()),
///     unit: Unit::Bohr,
///     ..Default::default()
/// }).unwrap();
/// assert_eq!(mol.natm, 2);
/// ```
#[allow(non_snake_case)]
pub fn M(args: MoleBuildArgs) -> Result<Mole, pyscf_core::PyscfRsError> {
    let mut mol = Mole::default();
    build_from(&mut mol, args)?;
    Ok(mol)
}

/// Populate a Mole from `MoleBuildArgs`.
///
/// Plan 02-02 wires:
///   - `format_atom` (GTO-01) — parsed `_atom`, `natm`.
///   - The scalar fields (`charge`, `spin`, `unit`, `verbose`, ...).
///   - `nelectron` computation (`sum(atom_charges) - charge`).
///
/// Plan 02-04 will extend this to also call `format_basis` + `make_env`
/// (the cintx flat-array projection). Plan 02-07 will extend with
/// `format_ecp` + `make_ecp_env`.
pub fn build_from(
    mol: &mut Mole,
    args: MoleBuildArgs,
) -> Result<(), pyscf_core::PyscfRsError> {
    // Echo user input for `dumps()` round-trip later.
    mol.atom = format!("{:?}", args.atom);
    mol.basis = format!("{:?}", args.basis);
    mol.ecp = format!("{:?}", args.ecp);

    // Scalar state.
    mol.charge = args.charge;
    mol.spin = args.spin;
    mol.cart = args.cart;
    mol.unit = args.unit;
    mol.verbose = args.verbose;
    mol.max_memory = args.max_memory;
    mol.output = args.output;
    mol.symmetry = false; // out of scope per ROADMAP.
    mol.groupname = "C1".to_string();
    mol.topgroup = "C1".to_string();

    // GTO-01: format_atom.
    let parsed_atoms =
        format_atom::format_atom(&args.atom, args.unit, args.origin, args.axes)?;
    mol.natm = parsed_atoms.len();
    mol._atom = parsed_atoms;

    // Compute nelectron from atom_charges - mol.charge (per upstream
    // tot_electrons at pyscf/gto/mole.py:1162-1186).
    let total_z: i32 = mol
        ._atom
        .iter()
        .filter_map(|(s, _)| format_atom::charge_for_symbol(s))
        .sum();
    let nelec = total_z - mol.charge;
    if nelec < 0 {
        return Err(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!(
                "negative electron count: {nelec} (total Z = {total_z}, charge = {})",
                mol.charge
            )),
        ));
    }
    mol.nelectron = nelec as usize;

    // Plan 02-04 will populate _atm/_bas/_env/ao_loc_nr/nao_nr/basis_set
    // here. For 02-02 those stay at Default::default() (empty Vecs / 0 / None).
    //
    // Mole is "built" only after plan 02-04 wires the basis projection.
    // Plan 02-02 leaves _built = false so calls to `mol.intor(...)` etc.
    // (when 02-05 lands them) fail early on an unbuilt Mole.
    mol._built = false;

    Ok(())
}

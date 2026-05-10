//! GTO-08 ≥30-attribute floor regression test.
//!
//! Compile-time guarantee that every field in the RESEARCH "Mole Attribute
//! Floor" table is present on `pyscf_core::Mole` (field-access compiles ⇒
//! field exists) plus runtime checks that the plan-02-02 portion of the
//! floor populates correctly.
//!
//! Plans 02-04 (basis projection), 02-07 (ECP loading), and 02-08 (`dumps`/
//! `loads`) extend this assertion as their own fields fill in.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};

#[test]
fn h2o_attribute_floor_present_and_defaults_sane() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0; H 0 0.7 0.6; H 0 -0.7 0.6".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .unwrap();

    // === All 33 floor items must be field-accessible (compile-time guard) ===
    let _ = &mol.atom;
    let _ = &mol.basis;
    let _ = &mol.ecp;
    let _ = mol.charge;
    let _ = mol.spin;
    let _ = mol.nelectron;
    let _ = mol.cart;
    let _ = mol.verbose;
    let _ = mol.max_memory;
    let _ = mol.unit;
    let _ = &mol.output;
    let _ = mol._built;
    let _ = mol.symmetry;
    let _ = &mol.groupname;
    let _ = &mol.topgroup;
    let _ = mol.natm;
    let _ = &mol._atom;
    let _ = mol.nbas;
    let _ = mol.nao_nr;
    let _ = mol.nao_2c;
    let _ = &mol.ao_loc_nr;
    let _ = &mol._basis;
    let _ = &mol._ecpbas;
    let _ = &mol._ecp;
    let _ = &mol._atm;
    let _ = &mol._bas;
    let _ = &mol._env;
    let _ = &mol.nucmod;
    let _ = &mol.nucprop;
    let _ = &mol.basis_set;
    let _ = &mol.pseudo;

    // Method-floor obligations:
    let _ = mol.atom_charges();
    let _ = mol.atom_coords();
    let _ = mol.atom_coord(0);
    let _ = mol.mass_list();
    let _ = mol.enuc();

    // === Runtime defaults sane ===
    assert_eq!(mol.natm, 3);
    assert_eq!(mol.nelectron, 8 + 1 + 1); // O + 2H = 10 e
    assert_eq!(mol.charge, 0);
    assert_eq!(mol.spin, 0);
    assert!(!mol.cart, "default = spherical AOs");
    assert!(!mol.symmetry, "v1 is C1 only");
    assert_eq!(mol.groupname, "C1");
    assert_eq!(mol.topgroup, "C1");
    assert_eq!(mol.unit, Unit::Bohr);

    // Plan 02-04 (not this plan) populates these — assert they are at
    // default for the 02-02 deliverable:
    assert!(mol._atm.is_empty(), "02-04 will populate _atm");
    assert!(mol._bas.is_empty(), "02-04 will populate _bas");
    assert!(mol._env.is_empty(), "02-04 will populate _env");
    assert!(
        mol.basis_set.is_none(),
        "02-04 will populate basis_set: Arc<BasisSet>"
    );
    assert!(
        !mol._built,
        "02-04 sets _built=true after the basis projection"
    );

    // enuc with 3 atoms in Bohr at the given positions:
    //   r_OH^2 = 0 + 0.7^2 + 0.6^2 = 0.85 → r_OH = sqrt(0.85)
    //   r_HH   = 1.4
    //   enuc = 8/sqrt(0.85) + 8/sqrt(0.85) + 1/1.4 = 16/sqrt(0.85) + 1/1.4
    let expected_enuc = 16.0 / 0.85_f64.sqrt() + 1.0 / 1.4;
    approx::assert_abs_diff_eq!(mol.enuc(), expected_enuc, epsilon = 1e-6);
}

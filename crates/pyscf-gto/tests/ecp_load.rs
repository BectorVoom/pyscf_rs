//! Plan 02-07 Task 2 acceptance tests: `format_ecp` + `make_ecp_env`
//! end-to-end. The LANL2DZ Cu fixture asserts loading correctness
//! against the actual file content of upstream
//! `pyscf/gto/basis/lanl2dz.dat` (Cu nelec 10, 3 channels: UL/S/P).
//!
//! NOTE on `n_core`: the plan text suggests `n_core == 18` for Cu in
//! LANL2DZ, but the actual upstream file ships
//! `Cu nelec 10` with three channels. We assert the actual file values
//! — see SUMMARY.md "Deviations from Plan" for the explanation.

use pyscf_gto::format_ecp::format_ecp;
use pyscf_gto::{AtomInput, BasisInput, EcpInput, MoleBuildArgs, M};
use pyscf_core::Unit;

fn cu_atom() -> Vec<(String, [f64; 3])> {
    vec![("Cu".to_string(), [0.0, 0.0, 0.0])]
}

#[test]
fn ecp_none_yields_empty_map() {
    let out = format_ecp(&EcpInput::None, &cu_atom()).unwrap();
    assert!(out.is_empty());
}

#[test]
fn ecp_lanl2dz_cu_loads_correctly_from_real_file() {
    // Asserts the values present in the actual upstream
    // `pyscf/gto/basis/lanl2dz.dat` Cu ECP block (nelec=10, 3 channels:
    // UL, S, P).
    let out = format_ecp(&EcpInput::Name("lanl2dz".into()), &cu_atom()).unwrap();
    let cu = out.get("CU").expect("Cu ECP entry expected");
    assert_eq!(
        cu.n_core, 10,
        "LANL2DZ Cu n_core (file: `Cu nelec 10`)",
    );
    assert!(
        cu.channels.len() >= 3,
        "LANL2DZ Cu has 3 channels (UL, S, P) per the upstream file; got {}",
        cu.channels.len(),
    );
    // First channel is UL (l = -1) per parser convention.
    assert_eq!(cu.channels[0].l, -1, "first channel should be UL/local (l=-1)");
}

#[test]
fn ecp_inline_text_h_ul_parses() {
    let text = "ECP\nH nelec 0\nH ul\n2 1.0 0.0\nEND";
    let out = format_ecp(
        &EcpInput::NwchemEcpText(text.into()),
        &[("H".into(), [0.0, 0.0, 0.0])],
    )
    .unwrap();
    assert!(out.contains_key("H"), "got map keys {:?}", out.keys().collect::<Vec<_>>());
    assert_eq!(out["H"].n_core, 0);
    assert_eq!(out["H"].channels.len(), 1);
    assert_eq!(out["H"].channels[0].l, -1, "ul = local channel = l = -1");
}

#[test]
fn ecp_per_element_routing() {
    use std::collections::HashMap;
    let mut m = HashMap::new();
    m.insert("Cu".to_string(), EcpInput::Name("lanl2dz".into()));
    let out = format_ecp(
        &EcpInput::PerElement(m),
        &[
            ("Cu".into(), [0.0, 0.0, 0.0]),
            ("H".into(), [0.0, 0.0, 1.0]),
        ],
    )
    .unwrap();
    assert!(out.contains_key("CU"), "Cu in PerElement should be loaded");
    assert!(
        !out.contains_key("H"),
        "H not in PerElement map → no ECP entry",
    );
}

#[test]
fn full_pipeline_cu_lanl2dz_via_M() {
    // End-to-end: M(...) builds Cu/lanl2dz; CHARGE_OF subtraction
    // happens; nelectron correct.
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("Cu 0 0 0".into()),
        basis: BasisInput::Name("lanl2dz".into()),
        ecp: EcpInput::Name("lanl2dz".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        ..Default::default()
    })
    .expect("build Cu/lanl2dz");

    // Cu = 29; LANL2DZ removes 10 core electrons → valence Z = 19.
    // _atm[CHARGE_OF] should be 19 (Pitfall 2: CHARGE_OF subtraction
    // happened exactly once).
    assert_eq!(
        mol._atm[pyscf_core::raw_layout::CHARGE_OF],
        19,
        "Cu(Z=29) − LANL2DZ(n_core=10) = 19 valence electrons in `_atm[CHARGE_OF]`",
    );
    assert_eq!(mol.nelectron, 19);
    // _ecpbas dimension sane (Pitfall 2 mitigation).
    assert_eq!(
        mol._ecpbas.len() % pyscf_core::raw_layout::BAS_SLOTS,
        0,
        "_ecpbas dim must be a multiple of BAS_SLOTS",
    );
    assert!(!mol._ecpbas.is_empty(), "_ecpbas should be populated");
    // _ecp HashMap populated.
    assert!(mol._ecp.contains_key("CU"));
}

#[test]
fn ecp_no_double_subtraction_on_repeat_build() {
    // Pitfall 2 defensive: build twice — CHARGE_OF must not be
    // subtracted twice.
    let args = MoleBuildArgs {
        atom: AtomInput::String("Cu 0 0 0".into()),
        basis: BasisInput::Name("lanl2dz".into()),
        ecp: EcpInput::Name("lanl2dz".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        ..Default::default()
    };
    let mol1 = M(args.clone()).expect("build mol1");
    let mol2 = M(args).expect("build mol2");
    assert_eq!(
        mol1._atm[pyscf_core::raw_layout::CHARGE_OF],
        mol2._atm[pyscf_core::raw_layout::CHARGE_OF],
        "two fresh builds must yield identical CHARGE_OF",
    );
    assert_eq!(
        mol1._atm[pyscf_core::raw_layout::CHARGE_OF],
        19,
        "Cu(Z=29) − LANL2DZ(n_core=10) = 19 (must not be 9 = 19 − 10 again)",
    );
}

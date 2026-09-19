use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
use pyscf_core::Unit;
#[test]
fn probe() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::Tuples(vec![("Ne".into(), [0.0, 0.0, 0.0])]),
        basis: BasisInput::Name("6-31g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    }).unwrap();
    let nbas = mol._bas.len() / 8;
    println!("nbas={nbas}");
    for i in 0..nbas {
        let b = &mol._bas[i*8..i*8+8];
        println!("shell {i}: l={} nprim={} nctr={} pexp={} pcoeff={}", b[1], b[2], b[3], b[5], b[6]);
    }
}

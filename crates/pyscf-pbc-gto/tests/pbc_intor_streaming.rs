//! The streaming lattice sum (`PYSCF_PBC_INTOR_IMAGE_CACHE=0`, or blocks too
//! large to cache): images are evaluated in waves and folded at once instead
//! of holding every image's blocks. Each output element receives its
//! additions in the same ascending-image order, so streaming is BIT-identical
//! to the cached route — checked with 3-image waves (several waves, a ragged
//! last one) for ovlp and kin, both fills, gamma and a general k-point.
//!
//! One test per binary: the switches are environment variables.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, PbcIntorOpts, pbc_intor};

fn tao_cell() -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("Ta".into(), [0.0, 0.0, 0.0]),
                ("O".into(), [1.95, 0.0, 0.0]),
            ]),
            basis: BasisInput::Name("gth-dzvp-molopt-sr".into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([[3.9, 0.0, 0.0], [0.0, 3.9, 0.0], [0.0, 0.0, 3.9]]),
        pseudo: Some("gth-pbe".into()),
        ..Default::default()
    })
    .expect("TaO cell must build")
}

#[test]
fn streamed_lattice_sum_is_bit_identical_to_cached() {
    let cell = tao_cell();
    let kpts = [[0.0; 3], [0.13, -0.21, 0.05]];
    let mut checked = 0;
    for intor in ["int1e_ovlp", "int1e_kin"] {
        for hermi in [0, 1] {
            let opts = || PbcIntorOpts { hermi, ..Default::default() };
            // SAFETY: the only test in this binary.
            unsafe {
                std::env::remove_var("PYSCF_PBC_INTOR_IMAGE_CACHE");
                std::env::remove_var("PYSCF_PBC_INTOR_WAVE_IMAGES");
            }
            let cached = pbc_intor(&cell, intor, &kpts, opts()).expect("cached");
            unsafe {
                std::env::set_var("PYSCF_PBC_INTOR_IMAGE_CACHE", "0");
                std::env::set_var("PYSCF_PBC_INTOR_WAVE_IMAGES", "3");
            }
            let streamed = pbc_intor(&cell, intor, &kpts, opts()).expect("streamed");
            assert_eq!(cached.kmats.len(), streamed.kmats.len());
            for (a, b) in cached.kmats.iter().zip(&streamed.kmats) {
                let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&a.re), bits(&b.re), "{intor} hermi={hermi}: real plane differs");
                assert_eq!(bits(&a.im), bits(&b.im), "{intor} hermi={hermi}: imaginary plane differs");
                assert!(a.re.iter().any(|v| v.abs() > 0.1), "vacuous matrix");
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 4);
}

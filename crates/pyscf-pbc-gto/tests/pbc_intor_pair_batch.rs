//! `PYSCF_PBC_INTOR_PAIR_BATCH=1` — the lattice sum's `int1e_ovlp_sph` /
//! `int1e_kin_sph` shell pairs evaluated as ONE cintx `PairBatchRequest` per
//! image instead of one `SessionRequest` per pair. The batched kernel is a
//! different code path, so it is gated against the scalar route at rounding
//! level, on a cell with the shells a DZVP transition-metal run has (Ta: s, p,
//! d, f; O: s, p, d) and on both fills (`hermi` 0 and 1), screened and not.
//!
//! One test per binary: the switch is an environment variable, and the
//! image-block cache is turned off so the two routes cannot share blocks.

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
fn batched_pairs_match_scalar_pairs() {
    // SAFETY: the only test in this binary.
    unsafe { std::env::set_var("PYSCF_PBC_INTOR_IMAGE_CACHE", "0") };
    let cell = tao_cell();
    let kpts = [[0.0; 3], [0.13, -0.21, 0.05]];
    let mut checked = 0;
    for intor in ["int1e_ovlp", "int1e_kin"] {
        for (hermi, screen) in [(0, false), (1, false), (1, true)] {
            let opts = || PbcIntorOpts { hermi, screen, ..Default::default() };
            unsafe { std::env::remove_var("PYSCF_PBC_INTOR_PAIR_BATCH") };
            let t = std::time::Instant::now();
            let scalar = pbc_intor(&cell, intor, &kpts, opts()).expect("scalar");
            let t_scalar = t.elapsed().as_secs_f64();
            unsafe { std::env::set_var("PYSCF_PBC_INTOR_PAIR_BATCH", "1") };
            let t = std::time::Instant::now();
            let batched = pbc_intor(&cell, intor, &kpts, opts()).expect("batched");
            let t_batched = t.elapsed().as_secs_f64();
            unsafe { std::env::remove_var("PYSCF_PBC_INTOR_PAIR_BATCH") };

            assert_eq!(scalar.kmats.len(), batched.kmats.len());
            let (mut worst, mut scale) = (0.0_f64, 0.0_f64);
            for (a, b) in scalar.kmats.iter().zip(&batched.kmats) {
                assert_eq!(a.re.len(), b.re.len());
                for t in 0..a.re.len() {
                    worst = worst.max((a.re[t] - b.re[t]).abs()).max((a.im[t] - b.im[t]).abs());
                    scale = scale.max(a.re[t].abs());
                }
            }
            assert!(scale > 0.1, "{intor}: vacuous matrix (max {scale:e})");
            assert!(
                worst <= 1e-12 * scale,
                "{intor} hermi={hermi} screen={screen}: batched differs by {worst:e} (scale {scale:e})"
            );
            eprintln!(
                "{intor} hermi={hermi} screen={screen}: max |Δ| {worst:e} (scale {scale:.3e}); \
                 scalar {t_scalar:.2}s, batched {t_batched:.2}s"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 6);
}

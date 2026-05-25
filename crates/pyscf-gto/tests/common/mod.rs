//! Shared test fixture helpers. Plans 02-02..02-08 extend.
//!
//! Wave 0 ships an inline H2/STO-3G `BasisSet` constructor for the cintx
//! round-trip smoke test (W0-T1).

use cintx_core::{Atom, BasisSet, NuclearModel, Representation, Shell, ShellTuple};
use std::sync::Arc;

/// Construct a minimal H2 BasisSet at STO-3G (spherical, 2 shells, 2 AOs).
///
/// Bond length 1.4 Bohr (≈ equilibrium for H2).
/// STO-3G H 1s primitive parameters from Hehre, Stewart & Pople
/// (J. Chem. Phys. 51, 2657, 1969):
///   exponents:    [3.42525091, 0.62391373, 0.16885540]
///   coefficients: [0.15432897, 0.53532814, 0.44463454]
///
/// Returns `(basis, shells)` where `shells` is the 2-tuple covering both H 1s
/// shells (suitable for a 2-center 1e overlap session).
// Shared test helper; not every test binary including this module uses it.
#[allow(dead_code)]
#[allow(clippy::type_complexity)]
pub fn h2_sto3g_basis() -> (BasisSet, ShellTuple) {
    // STO-3G H 1s (port from cintx oracle test build_h2o_sto3g):
    let h_1s_exp: Arc<[f64]> =
        Arc::from(vec![3.42525091_f64, 0.62391373, 0.16885540].into_boxed_slice());
    let h_1s_coeff: Arc<[f64]> =
        Arc::from(vec![0.15432897_f64, 0.53532814, 0.44463454].into_boxed_slice());

    // Two H atoms separated by 1.4 Bohr along z.
    let h1 = Atom::try_new(1, [0.0, 0.0, 0.0], NuclearModel::Point, None, None)
        .expect("H1 atom should validate");
    let h2 = Atom::try_new(1, [0.0, 0.0, 1.4], NuclearModel::Point, None, None)
        .expect("H2 atom should validate");
    let atoms: Arc<[Atom]> = Arc::from(vec![h1, h2].into_boxed_slice());

    // One s-shell per atom (l=0, 3 primitives, 1 contraction, spherical).
    let shell_h1 = Arc::new(
        Shell::try_new(
            0, // atom_index = 0 (H1)
            0, // ang_momentum = 0 (s)
            3, // nprim
            1, // nctr
            0, // kappa = 0 (use C/sph)
            Representation::Spheric,
            Arc::clone(&h_1s_exp),
            Arc::clone(&h_1s_coeff),
        )
        .expect("H1 1s shell should validate"),
    );
    let shell_h2 = Arc::new(
        Shell::try_new(
            1, // atom_index = 1 (H2)
            0,
            3,
            1,
            0,
            Representation::Spheric,
            Arc::clone(&h_1s_exp),
            Arc::clone(&h_1s_coeff),
        )
        .expect("H2 1s shell should validate"),
    );

    let shells_arc: Arc<[Arc<Shell>]> =
        Arc::from(vec![Arc::clone(&shell_h1), Arc::clone(&shell_h2)].into_boxed_slice());
    let basis = BasisSet::try_new(atoms, shells_arc).expect("H2/STO-3G basis should validate");
    let shell_tuple = ShellTuple::try_from_iter([shell_h1, shell_h2])
        .expect("2-shell tuple is within libcint arity (4)");

    (basis, shell_tuple)
}

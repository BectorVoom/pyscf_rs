//! F-03 SCAFFOLD — spinor (relativistic 2-component) integral representation.
//!
//! This module is the **planned-feature scaffold** for `intor(..._spinor)`,
//! tracked in `.planning/features/F-03-spinor-integrals-PLAN.md`. It is NOT a
//! working implementation: the complex `sph2spinor` transform and the
//! `intor_spinor` entry point are honest stubs that return
//! `NotYetImplemented` until the planned tasks land *with a live-PySCF byte
//! identity oracle*. Only the pieces that are pure and verifiable in-sandbox
//! today are implemented here (the 2-component AO count formula and the
//! complex output container).
//!
//! ## Why spinor integrals need their own surface
//!
//! Spinor integrals are **complex-valued**: the spherical→spinor transform
//! couples the `2(2l+1)` real spherical harmonics × {α,β} spin functions into
//! the `j = l ± 1/2` adapted basis using complex Clebsch–Gordan coefficients.
//! The non-relativistic `IntorOutput` carries `Vec<f64>` (real only), so the
//! spinor path returns a distinct [`IntorOutputComplex`] rather than mutating
//! the real type that scf/mp2/df/PyO3 all consume.
//!
//! Provenance for the eventual transform: libcint `c2s.c CINTc2s_bra_spinor`
//! and PySCF `pyscf/gto/mole.py sph2spinor_coeff` (Apache-2.0).

use crate::layout_table::IntorLayout;
use pyscf_core::{Mole, PyscfRsError};

/// Complex integral output — the spinor analogue of [`crate::IntorOutput`].
///
/// `re`/`im` are parallel flat F-order buffers (same length, same logical
/// `shape`), so element `k` of the matrix is `re[k] + i·im[k]`. Kept as two
/// `f64` buffers (rather than pulling in a `num_complex` dependency) so the
/// type is self-contained and trivially FFI/serialise-friendly.
#[derive(Debug, Clone)]
pub struct IntorOutputComplex {
    /// Real part, flat F-order. `re.len() == shape.iter().product()`.
    pub re: Vec<f64>,
    /// Imaginary part, flat F-order. `im.len() == re.len()`.
    pub im: Vec<f64>,
    /// Logical shape; for an arity-2 spinor integral this is `[n2c, n2c]`.
    pub shape: Vec<usize>,
    /// Per-intor layout convention (Pitfall 8), shared with the real path.
    pub layout: IntorLayout,
}

impl IntorOutputComplex {
    /// Number of complex elements (`re.len()`), i.e. `shape.iter().product()`.
    pub fn len(&self) -> usize {
        self.re.len()
    }

    /// True when the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.re.is_empty()
    }
}

/// Number of 2-component spinor AOs contributed by a single shell of angular
/// momentum `l` with `nctr` contractions.
///
/// A shell of angular momentum `l` splits into `j = l − 1/2` (`2j+1 = 2l`
/// states) and `j = l + 1/2` (`2j+1 = 2l+2` states); the total per
/// contraction is `2l + (2l+2) = 2(2l+1)` — exactly twice the spherical count
/// `(2l+1)`. Hence the whole-molecule relation `n2c = 2 · nao_sph`.
///
/// This is pure arithmetic and fully unit-tested below; it is the building
/// block the planned `mol.nao_2c` wiring (task T1) will sum over shells.
pub fn n2c_per_shell(l: u32, nctr: usize) -> usize {
    nctr * 2 * (2 * l as usize + 1)
}

/// SCAFFOLD STUB — spinor entry point. Returns [`IntorOutputComplex`] once the
/// planned tasks land; today returns `NotYetImplemented`.
///
/// The split from [`crate::intor`] is deliberate: `intor` returns the real
/// `IntorOutput`, spinor integrals are complex, so they get a distinct
/// signature rather than overloading the real type. See
/// `.planning/features/F-03-spinor-integrals-PLAN.md` for the task DAG and the
/// live-PySCF verification gate.
pub fn intor_spinor(mol: &Mole, name: &str) -> Result<IntorOutputComplex, PyscfRsError> {
    let _ = (mol, name);
    Err(PyscfRsError::NotYetImplemented {
        phase: 3,
        what: "intor_spinor: spinor representation is scaffolded only \
               (see .planning/features/F-03-spinor-integrals-PLAN.md) — \
               complex output type ready; sph2spinor transform + cintx spinor \
               operators + live-PySCF oracle are the remaining planned tasks",
    })
}

/// SCAFFOLD STUB — single `sph2spinor` Clebsch–Gordan coefficient
/// `U[l][spinor_row][(sph_col, spin)]` (complex).
///
/// Returns `(re, im)` once the coefficient tables land. Today this is the
/// explicit gap: the tables must be transcribed from libcint
/// `CINTc2s_bra_spinor` / PySCF `sph2spinor_coeff` and validated byte-for-byte
/// against live PySCF (no oracle-free invariant is strong enough — Hermiticity
/// and shape pass for wrong phase/ordering conventions). See plan task T2.
pub fn sph2spinor_coeff(
    l: u32,
    spinor_row: usize,
    sph_col: usize,
    spin: usize,
) -> Result<(f64, f64), PyscfRsError> {
    let _ = (l, spinor_row, sph_col, spin);
    Err(PyscfRsError::NotYetImplemented {
        phase: 3,
        what: "sph2spinor_coeff: CG coefficient table not yet transcribed \
               (plan task T2) — requires libcint CINTc2s_bra_spinor provenance \
               + live-PySCF byte-identity validation",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Verifiable-today scaffold: the AO-count formula ──────────────────

    /// `n2c_per_shell` against hand-computed values for s/p/d/f and the
    /// `n2c = 2·(2l+1)·nctr` relation. Pure arithmetic — no oracle needed.
    #[test]
    fn n2c_per_shell_matches_2_times_spherical() {
        // (l, nctr) → expected 2-component count.
        assert_eq!(n2c_per_shell(0, 1), 2); // s: 2·1
        assert_eq!(n2c_per_shell(1, 1), 6); // p: 2·3
        assert_eq!(n2c_per_shell(2, 1), 10); // d: 2·5
        assert_eq!(n2c_per_shell(3, 1), 14); // f: 2·7
        // nctr scaling + the 2·nsph relation.
        for l in 0..=6u32 {
            for nctr in 0..=4usize {
                let nsph = nctr * (2 * l as usize + 1);
                assert_eq!(n2c_per_shell(l, nctr), 2 * nsph, "l={l} nctr={nctr}");
            }
        }
    }

    /// The complex container's `len`/`is_empty` track the real buffer.
    #[test]
    fn complex_output_len_tracks_buffers() {
        let out = IntorOutputComplex {
            re: vec![1.0, 2.0, 3.0, 4.0],
            im: vec![0.0, -1.0, 0.0, 1.0],
            shape: vec![2, 2],
            layout: IntorLayout::ScalarFOrder,
        };
        assert_eq!(out.len(), 4);
        assert_eq!(out.len(), out.shape.iter().product::<usize>());
        assert!(!out.is_empty());
    }

    // ── Stubs must error cleanly (never panic, never fake numbers) ────────

    #[test]
    fn intor_spinor_is_scaffold_stub() {
        // A default (unbuilt) Mole is fine — the stub errors before touching it.
        let mol = Mole::default();
        let r = intor_spinor(&mol, "int1e_ovlp_spinor");
        assert!(
            matches!(r, Err(PyscfRsError::NotYetImplemented { phase: 3, .. })),
            "expected scaffold NotYetImplemented, got {r:?}"
        );
    }

    #[test]
    fn sph2spinor_coeff_is_scaffold_stub() {
        let r = sph2spinor_coeff(1, 0, 0, 0);
        assert!(
            matches!(r, Err(PyscfRsError::NotYetImplemented { phase: 3, .. })),
            "expected scaffold NotYetImplemented, got {r:?}"
        );
    }

    // ── Oracle contract (planned) ────────────────────────────────────────
    // These encode the verification gate the implementation MUST pass. They
    // are #[ignore]'d until the live-PySCF oracle harness + the transform land
    // (plan tasks T2–T4); running them before then would assert on a stub.

    #[test]
    #[ignore = "F-03 task T4: requires sph2spinor transform + live-PySCF oracle"]
    fn int1e_ovlp_spinor_byte_matches_upstream() {
        // CONTRACT: intor_spinor(H2O/cc-pVDZ, "int1e_ovlp_spinor") must agree
        // with upstream mol.intor("int1e_ovlp_spinor") (complex) to atol=1e-10,
        // shape [n2c, n2c] with n2c = 2·nao_nr.
        unimplemented!("oracle harness pending — see F-03 plan T4");
    }

    #[test]
    #[ignore = "F-03 task T4: spinor overlap must be Hermitian (necessary, not sufficient)"]
    fn int1e_ovlp_spinor_is_hermitian() {
        // CONTRACT (weak invariant, kept as a secondary guard): S = S†, i.e.
        // re symmetric and im antisymmetric. NOTE: passing this does NOT prove
        // correctness — it is why T4 also requires the byte-identity oracle.
        unimplemented!("pending intor_spinor implementation — see F-03 plan T4");
    }
}

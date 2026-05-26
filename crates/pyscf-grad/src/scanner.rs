//! `GradScanner` — the geometry-optimization seam (07-04 consumer).
//!
//! Source: `pyscf/grad/rhf.py:248-262` `SCF_GradScanner.__call__`:
//! ```python
//! def __call__(self, mol_or_geom, **kwargs):
//!     mol = ...                       # set the new geometry
//!     self.reset(mol)
//!     e_tot = self.base(mol)          # the as_scanner energy (SCF-12/MP2-07/CCSD-07)
//!     de    = self.kernel(**kwargs)   # the analytical gradient
//!     return e_tot, de                # ← what the optimizer consumes
//! ```
//!
//! Structural analog: `crates/pyscf-scf/src/scanner.rs` — the
//! `Box<dyn Fn(&Mole) -> Result<Energy, PyscfRsError> + Send + Sync>` closure
//! that captures scalar settings + re-runs `kernel()` on a new `Mole`. The
//! GRAD scanner wraps that ENERGY scanner (captured by value) + appends a
//! gradient `kernel` closure, returning `(Energy, de)` instead of `Energy` —
//! this is the single `Mole -> (e_tot, de)` seam the geometry optimizer (07-04)
//! consumes.
//!
//! The two closures are supplied by the per-method wave (07-03 wires RHF's
//! `as_scanner` energy + its `kernel`); this struct fixes the SEAM from day one
//! (D-09) so the optimizer wires against a stable shape.

use pyscf_core::{Energy, Mole, PyscfRsError};

/// A `Send + Sync` energy closure (the `base` / `as_scanner` seam): given a new
/// `Mole`, re-run the method's SCF/post-SCF and return the total energy.
pub type EnergyClosure = Box<dyn Fn(&Mole) -> Result<Energy, PyscfRsError> + Send + Sync>;

/// A `Send + Sync` gradient closure (the `kernel` seam): given a new `Mole` and
/// an optional `atmlst` subset, return the analytical gradient as one
/// `[x, y, z]` per requested atom (Hartree/Bohr).
pub type GradClosure =
    Box<dyn Fn(&Mole, Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> + Send + Sync>;

/// The geometry-optimization scanner: a `Mole -> (e_tot, de)` evaluator.
///
/// Mirrors `pyscf/grad/rhf.py:248-262`: on a new geometry it evaluates the
/// energy closure (`base`) then the gradient closure (`kernel`), returning the
/// `(e_tot, de)` tuple the optimizer (07-04) drives its line-search on. Both
/// closures capture their method's scalar settings by value (the
/// `pyscf-scf::as_scanner` discipline) so the scanner is `Send + Sync` and can
/// be moved across threads.
pub struct GradScanner {
    base: EnergyClosure,
    grad: GradClosure,
}

impl GradScanner {
    /// Build a scanner from an energy closure (the `as_scanner` seam) and a
    /// gradient closure (the `kernel` seam).
    pub fn new(base: EnergyClosure, grad: GradClosure) -> Self {
        Self { base, grad }
    }

    /// Evaluate `(e_tot, de)` at the given geometry (`rhf.py:248-262`): run the
    /// energy closure, then the gradient closure with the requested `atmlst`
    /// subset. The returned `de` has shape `(n, 3)` where `n = atmlst.len()`
    /// (or `mol.natm` when `atmlst` is `None`).
    pub fn eval(
        &self,
        mol: &Mole,
        atmlst: Option<&[usize]>,
    ) -> Result<(Energy, Vec<[f64; 3]>), PyscfRsError> {
        let e_tot = (self.base)(mol)?;
        let de = (self.grad)(mol, atmlst)?;
        Ok((e_tot, de))
    }
}

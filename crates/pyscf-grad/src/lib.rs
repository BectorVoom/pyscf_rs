//! pyscf-grad: Analytical gradients (HF / DFT / MP2 / CCSD) + the geomopt seam.
//!
//! Phase 7 deliverable. Ports upstream `pyscf/grad/*.py` file-for-file (sibling-
//! crate fidelity, mirroring the Phase-6 `pyscf-ccsd` discipline): the module
//! split, the error/hooks/scanner shapes, and the host-loop reduction
//! discipline all copy `pyscf-ccsd`. This crate stays strictly pyo3-free
//! (D-07/D-08 — the PyO3 bridge lives exclusively in `pyscf-py`), cubecl-free
//! (the algebra wall — every reduction/contraction routes through
//! `pyscf-algebra`), and owns no `hdf5-metno` dep. All reductions go through
//! `oracle_sum`/`oracle_dot` for bit-exactness under `release-oracle`; NO bare
//! `+=` in any gradient/CPHF/FD reduction.
//!
//! This plan (07-02) ships the crate skeleton + the four load-bearing contract
//! types every later wave consumes:
//!   - the base [`Gradients`] trait (mirroring `pyscf/grad/rhf.py:265-418`
//!     `GradientsBase`): the `mol`/`base`/`unit`/`atmlst`/`de` fields +
//!     `kernel`/`grad_elec`/`grad_nuc`/`make_rdm1e`/`hcore_generator`/`get_ovlp`,
//!     with `atmlst` row-subsetting (GRAD-08) built into `kernel` from day one
//!     (D-09);
//!   - [`GradError`] (the error bridge);
//!   - [`GradOverrideHooks`] + [`NoGradOverrides`] (the DF-grad / extra_force
//!     subclass-override seam);
//!   - [`GradScanner`] (the `Mole -> (e_tot, de)` geomopt seam, 07-04) +
//!     [`verify_fd`] (the D-01 always-on numeric gate, GRAD-09).
//!
//! Each method module (`rhf`/`uhf`/`rks`/`uks`/`mp2`/`ccsd`/`ecp`/`cphf`) is a
//! stub whose body returns `GradError::NotYetImplemented { wave }` (`wave` = the
//! Phase-7 wave that fills it); the bodies land in 07-03..07-08.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod ccsd;
pub mod cphf;
pub mod ecp;
pub mod error;
pub mod hooks;
pub mod mp2;
pub mod rhf;
pub mod rks;
pub mod scanner;
pub mod uhf;
pub mod uks;
pub mod verify_fd;

pub use error::GradError;
pub use hooks::{GradOverrideHooks, NoGradOverrides};
pub use scanner::GradScanner;
pub use verify_fd::{DEFAULT_DISP, FD_TOL, verify_fd};

use pyscf_core::{Mole, PyscfRsError, Unit};

/// The base gradient contract — the in-tree analog of
/// `pyscf/grad/rhf.py:265-418` `GradientsBase`.
///
/// Carries the five `GradientsBase` state fields (`mol`/`base`/`unit`/`atmlst`/
/// `de`) as accessor methods and the six method-API entry points
/// (`kernel`/`grad_elec`/`grad_nuc`/`make_rdm1e`/`hcore_generator`/`get_ovlp`).
/// Two of those (`grad_nuc`, `get_ovlp`) are SHARED trait-default methods —
/// every method inherits the same nuclear-repulsion gradient and overlap-
/// derivative; `grad_elec` and `make_rdm1e` are PER-METHOD (the trait requires
/// them, each RHF/UHF/RKS/... impl supplies the body).
///
/// `kernel` assembles `de = grad_elec(atmlst) + grad_nuc(atmlst)` and stores it,
/// applying `atmlst` ROW-SUBSETTING (GRAD-08) from day one (D-09): when
/// `atmlst = Some(&[i, j, ..])` the returned `de` contains EXACTLY those atom
/// rows (shape `(atmlst.len(), 3)`); when `None`, all `natm` rows.
///
/// The whole-crate FD self-verification gate ([`verify_fd`], GRAD-09, D-01) is
/// a free function rather than a trait method because it is reduction-discipline
/// machinery shared by every method — but it is available "on the base API from
/// day one" in the D-09 sense: every method's `as_scanner` energy + `kernel`
/// gradient feed straight into it.
pub trait Gradients {
    /// The molecule whose geometry the gradient is taken with respect to.
    /// (`GradientsBase.mol`.)
    fn mol(&self) -> &Mole;

    /// The atom subset the gradient is restricted to (`GradientsBase.atmlst`):
    /// `Some(&[i, j, ..])` restricts `kernel` to those atom rows; `None` means
    /// the full molecule. Bounds-checked against `natm` in [`Gradients::kernel`]
    /// (T-07-06).
    fn atmlst(&self) -> Option<&[usize]>;

    /// The last computed gradient `(n, 3)` flattened row-major (`GradientsBase.de`),
    /// where `n = atmlst.len()` (or `natm` when `atmlst` is `None`). `None` until
    /// [`Gradients::kernel`] has run.
    fn de(&self) -> Option<&[[f64; 3]]>;

    /// The coordinate unit the caller works in (`GradientsBase.unit`). The
    /// gradient is always returned in Hartree/Bohr regardless (the FD gate
    /// displaces in Bohr); `unit` only affects how a driver reports the step.
    fn unit(&self) -> Unit {
        Unit::Bohr
    }

    /// The electronic (method-specific) gradient `dE_elec/dR`, restricted to
    /// the `atmlst` rows. PER-METHOD: each RHF/UHF/RKS/... impl supplies the
    /// Hellmann-Feynman + Pulay assembly. Returns one `[x, y, z]` per requested
    /// atom (in the `atmlst` order).
    fn grad_elec(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError>;

    /// The energy-weighted reduced density matrix (`rhf.py:185-189`,
    /// `make_rdm1e`). PER-METHOD — it feeds the overlap-Pulay term in
    /// `grad_elec`. Returns the flattened `(nao, nao)` matrix.
    fn make_rdm1e(&self) -> Result<Vec<f64>, PyscfRsError>;

    /// The nuclear-repulsion gradient `dE_nuc/dR` (`rhf.py:92-107`), restricted
    /// to the `atmlst` rows. SHARED trait-default: a thin Coulomb-force port
    /// over the already-shipped nuclear charges + coordinates, identical for
    /// every method. Materialise-then-`oracle_sum` per pair (Pitfall 2) — NO
    /// bare `+=`.
    fn grad_nuc(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        let mol = self.mol();
        let coords = mol.atom_coords();
        let charges = mol.atom_charges();
        let natm = mol.natm;
        let rows = resolve_atmlst(atmlst, natm)?;
        let mut de = Vec::with_capacity(rows.len());
        for &ia in &rows {
            // Accumulate the per-component force from every other nucleus into
            // a materialised buffer, then ordered-sum it (no bare `+=`).
            let (mut fx, mut fy, mut fz) = (Vec::new(), Vec::new(), Vec::new());
            for jb in 0..natm {
                if jb == ia {
                    continue;
                }
                let dx = coords[ia][0] - coords[jb][0];
                let dy = coords[ia][1] - coords[jb][1];
                let dz = coords[ia][2] - coords[jb][2];
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 <= 0.0 {
                    continue;
                }
                let r = r2.sqrt();
                let zz = (charges[ia] as f64) * (charges[jb] as f64);
                // dE_nuc/dR_ia = -sum_{j != i} Z_i Z_j (R_i - R_j) / |R_i - R_j|^3
                let inv_r3 = 1.0 / (r * r2);
                fx.push(-zz * dx * inv_r3);
                fy.push(-zz * dy * inv_r3);
                fz.push(-zz * dz * inv_r3);
            }
            de.push([
                pyscf_algebra::oracle_sum(&fx),
                pyscf_algebra::oracle_sum(&fy),
                pyscf_algebra::oracle_sum(&fz),
            ]);
        }
        Ok(de)
    }

    /// The core-Hamiltonian derivative generator (`rhf.py:121-143`,
    /// `hcore_generator`). SHARED-by-default but a `NotYetImplemented { wave: 3 }`
    /// seam this plan: it needs `int1e_iprinv` + `with_rinv_at_nucleus`, a
    /// per-atom origin shift ABSENT from cintx today (07-RESEARCH Open Q2). The
    /// RHF wave (07-03) supplies the body once the cintx workstream lands.
    fn hcore_generator(&self) -> Result<(), PyscfRsError> {
        Err(error::GradError::NotYetImplemented { wave: 3 }.into())
    }

    /// The overlap derivative `-int1e_ipovlp` (`rhf.py:146`, `get_ovlp`).
    /// SHARED-by-default but a `NotYetImplemented { wave: 3 }` seam this plan
    /// (it needs `int1e_ipovlp`, a Phase-7 cintx workstream item). The RHF wave
    /// supplies the body.
    fn get_ovlp(&self) -> Result<Vec<f64>, PyscfRsError> {
        Err(error::GradError::NotYetImplemented { wave: 3 }.into())
    }

    /// Compute the full gradient: `de = grad_elec(atmlst) + grad_nuc(atmlst)`
    /// (`rhf.py:265-418` `GradientsBase.kernel`), applying `atmlst` ROW-
    /// SUBSETTING (GRAD-08) — when `atmlst = Some(&[i, j])` the returned `de`
    /// is exactly rows `[i, j]` of the full gradient (shape `(2, 3)`); when
    /// `None`, all `natm` rows. Out-of-range indices are bounds-checked
    /// (T-07-06) and return `GradError::ShapeMismatch`, never an OOB panic.
    ///
    /// Both `grad_elec` and `grad_nuc` receive the SAME `atmlst`, so they
    /// return rows in the same order; this method sums them element-wise
    /// through `oracle_sum` (no bare `+=`).
    fn kernel(&self, atmlst: Option<&[usize]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        // Validate atmlst once, up front (T-07-06) so both sub-gradients agree
        // on row count + order.
        let rows = resolve_atmlst(atmlst, self.mol().natm)?;
        let elec = self.grad_elec(atmlst)?;
        let nuc = self.grad_nuc(atmlst)?;
        if elec.len() != rows.len() || nuc.len() != rows.len() {
            return Err(error::GradError::ShapeMismatch {
                expected: rows.len(),
                got: elec.len().min(nuc.len()),
            }
            .into());
        }
        let mut de = Vec::with_capacity(rows.len());
        for k in 0..rows.len() {
            de.push([
                pyscf_algebra::oracle_sum(&[elec[k][0], nuc[k][0]]),
                pyscf_algebra::oracle_sum(&[elec[k][1], nuc[k][1]]),
                pyscf_algebra::oracle_sum(&[elec[k][2], nuc[k][2]]),
            ]);
        }
        Ok(de)
    }
}

/// Resolve a caller-supplied `atmlst` into a concrete, bounds-checked row list
/// (GRAD-08 + T-07-06). `None` → `0..natm`; `Some(&[..])` → those indices, each
/// validated `< natm` (an out-of-range index returns
/// `GradError::ShapeMismatch { expected: natm, got: idx }` rather than panicking
/// or indexing out of bounds — `#![forbid(unsafe_code)]` means a logic error is
/// never UB).
pub fn resolve_atmlst(atmlst: Option<&[usize]>, natm: usize) -> Result<Vec<usize>, GradError> {
    match atmlst {
        None => Ok((0..natm).collect()),
        Some(list) => {
            for &idx in list {
                if idx >= natm {
                    return Err(GradError::ShapeMismatch {
                        expected: natm,
                        got: idx,
                    });
                }
            }
            Ok(list.to_vec())
        }
    }
}

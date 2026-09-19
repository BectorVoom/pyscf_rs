//! `SCF_GradScanner` — `pyscf/pbc/grad/krhf.py:300-343` (18-19, Task 2).
//!
//! The load-bearing seam of the phase: 18-02's [`crate::verify_fd`] consumes
//! the energy half, and 18-14's optimizer drives the `(e_tot, de)` whole.
//!
//! # Upstream correspondence (`krhf.py` line → here)
//!
//! | upstream | here |
//! |---|---|
//! | `as_scanner(mf_grad)` `:300-320` (wrap unless already a scanner) | [`as_scanner`] (construction is explicit; see below) |
//! | `SCF_GradScanner.__call__` `:326-342` (`mf_scanner(cell)`, `self.cell = cell`, grids reset, `kernel`) | [`ScfGradScanner::scan`] |
//! | `:311-313` side-effect contract | [`ScfGradScanner`] (shared state) |
//! | `:338-339` `grids.reset` hook (KRKS subclass, 18-07) | [`ScfGradScanner::reset_grids`] (inert) |
//!
//! # Side-effect contract (`:311-313`)
//!
//! *"Scanner has side effects. It may change many underlying objects."* The
//! scanner owns shared mutable state ([`ScannerState`], behind `Arc<Mutex<…>>`):
//! every [`ScfGradScanner::scan`] re-runs KRHF on the new [`Cell`], stores the
//! converged density as the next call's initial guess (*"automatically use the
//! results of last calculation as the initial guess"*, `:306-309`), and
//! records the cell it last evaluated. Clones share the state — exactly the
//! aliasing upstream's in-place mutation has — and [`ScfGradScanner::energy_scanner`]
//! hands the same state to [`crate::verify_fd`] as an `Fn` closure.
//!
//! The `isinstance(mf_grad, lib.GradScanner)` early return (`:314-315`, `:240-241`
//! on the molecular side) has no Rust analog: construction is explicit through
//! [`as_scanner`]/[`ScfGradScanner::new`], so a scanner is never wrapped twice.
//!
//! # k-points and mesh on displaced cells
//!
//! The captured k-points are absolute-Cartesian and are reused verbatim on every
//! call. That is correct for nuclear displacements — which preserve the lattice
//! (18-CONTEXT trap 5's analog) — and [`crate::verify_fd`]'s displaced cells
//! additionally pin the FFT mesh on both sides, so neither the k-mesh nor the
//! mesh is ever differentiated. A strained cell (different lattice) needs a new
//! scanner with re-derived k-points, just as upstream's `reset(cell)` path
//! re-derives them.
//!
//! # Reductions and kernels (ALG-06, D-PBC-17)
//!
//! Every reduction routes through `pyscf_algebra::oracle_sum` over a
//! materialised partial buffer — never a bare `+=` — so results are
//! bit-identical for any `RAYON_NUM_THREADS`.
//!
//! No CubeCL kernel is added here (ALG-06: `pyscf-pbc-grad` may not depend on
//! `cubecl-*` AT ALL; `xtask check-dependency-wall` enforces it). The CubeCL
//! manual (`manual/Cubecl/INDEX.md`: generics-`Float` kernels, algebra/dot and
//! reduction manuals) was read before writing; there is no device kernel in
//! this file for it to apply to. Device computation is reached only through the
//! already-landed gradient assembly in [`crate::krhf`].

use std::sync::{Arc, Mutex};

use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_gto::{Cell, ExxDiv};
use pyscf_pbc_scf::{
    Krhf,
    types::{KDms, KInitGuess, KScfConfig},
};

use crate::gradients::{EnergyScanner, Gradient};
use crate::krhf::KrhfGradients;

fn invalid(message: impl Into<String>) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(message.into()))
}

fn locked<'a>(
    state: &'a Mutex<ScannerState>,
    what: &'static str,
) -> Result<std::sync::MutexGuard<'a, ScannerState>, PyscfRsError> {
    state
        .lock()
        .map_err(|_| invalid(format!("KRHF scanner: {what} state lock was poisoned")))
}

/// Scalar SCF settings captured by value (the `as_scanner` discipline: DIIS,
/// `conv_tol` and friends travel with the scanner, `krhf.py:306-309`).
#[derive(Debug, Clone)]
pub struct KrhfScannerConfig {
    /// Absolute-Cartesian sampling k-points, reused verbatim on every call.
    pub kpts: Vec<[f64; 3]>,
    /// Exchange-divergence treatment (`None` = no correction).
    pub exxdiv: Option<ExxDiv>,
    /// Energy convergence threshold.
    pub conv_tol: f64,
    /// Orbital-gradient threshold (`None` = `sqrt(conv_tol)`, upstream's rule).
    pub conv_tol_grad: Option<f64>,
    /// Maximum SCF cycles per scan.
    pub max_cycle: u32,
}

impl KrhfScannerConfig {
    /// Tight settings matching upstream's own gradient tests
    /// (`test_krhf.py:48`: `conv_tol=1e-10, conv_tol_grad=1e-6`).
    pub fn tight(kpts: &[[f64; 3]], exxdiv: Option<ExxDiv>) -> Self {
        Self {
            kpts: kpts.to_vec(),
            exxdiv,
            conv_tol: 1e-10,
            conv_tol_grad: Some(1e-6),
            max_cycle: 100,
        }
    }

    fn scf_config(&self, init_guess: KInitGuess) -> KScfConfig {
        KScfConfig {
            conv_tol: self.conv_tol,
            conv_tol_grad: self.conv_tol_grad,
            max_cycle: self.max_cycle,
            init_guess,
            ..KScfConfig::default()
        }
    }
}

/// The shared mutable state behind [`ScfGradScanner`].
#[derive(Debug, Clone, Default)]
struct ScannerState {
    /// The cell the scanner last evaluated (`self.cell = cell`, `:334`).
    cell: Option<Cell>,
    /// The last converged density, reused as the next call's initial guess.
    last_dm: Option<KDms>,
    /// Whether the last scan converged.
    last_converged: Option<bool>,
}

/// `SCF_GradScanner` — `krhf.py:322-342`.
///
/// A `Cell -> (e_tot, de)` evaluator over KRHF. Clones share one [`ScannerState`],
/// so the side-effect contract (`:311-313`) holds across handles, and the
/// energy seam handed to [`crate::verify_fd`] sees the same guess reuse as
/// [`ScfGradScanner::scan`].
#[derive(Debug, Clone)]
pub struct ScfGradScanner {
    config: KrhfScannerConfig,
    state: Arc<Mutex<ScannerState>>,
}

impl ScfGradScanner {
    /// Build a scanner over `cell` with `config` captured by value.
    ///
    /// An empty `kpts` normalises to the single gamma point (the `pbc_intor`
    /// convention [`crate::krhf`] uses everywhere).
    pub fn new(cell: Cell, mut config: KrhfScannerConfig) -> Self {
        if config.kpts.is_empty() {
            config.kpts = vec![[0.0; 3]];
        }
        Self {
            config,
            state: Arc::new(Mutex::new(ScannerState {
                cell: Some(cell),
                last_dm: None,
                last_converged: None,
            })),
        }
    }

    /// The sampling k-points (absolute-Cartesian, as captured).
    pub fn kpts(&self) -> Vec<[f64; 3]> {
        self.config.kpts.clone()
    }

    /// The exchange-divergence treatment (as captured).
    pub fn exxdiv(&self) -> Option<ExxDiv> {
        self.config.exxdiv
    }

    /// The cell last evaluated (the `:334` side effect), if any scan ran.
    pub fn cell(&self) -> Option<Cell> {
        locked(&self.state, "read")
            .map(|state| state.cell.clone())
            .unwrap_or(None)
    }

    /// Whether the last scan converged (`None` before the first scan).
    pub fn converged(&self) -> Option<bool> {
        locked(&self.state, "read")
            .map(|state| state.last_converged)
            .unwrap_or(None)
    }

    /// The `:338-339` grids hook, inert on the KRHF base class.
    ///
    /// Upstream resets `self.grids` here *only when the object has one* — the
    /// hook exists for the KRKS subclass (18-07), which carries a second
    /// integration grid. KRHF has no `grids` attribute, so this is a no-op by
    /// construction; it is declared (rather than omitted) so the 18-07
    /// subclass has the override point upstream gives it.
    pub fn reset_grids(&self, _cell: &Cell) {}

    /// The initial guess for a scan on `cell`: the stored density when its
    /// shape still fits (`nset = 1`, per-k `nao × nao`), else MINAO.
    fn guess_for(&self, state: &ScannerState, nao: usize, nkpts: usize) -> KInitGuess {
        match &state.last_dm {
            Some(dm)
                if dm.len() == 1
                    && dm[0].len() == nkpts
                    && dm[0].iter().all(|m| {
                        m.re.len() == nao * nao
                            && m.im.len() == nao * nao
                            && m.re.iter().chain(&m.im).all(|v| v.is_finite())
                    }) =>
            {
                KInitGuess::UserDm(dm.clone())
            }
            _ => KInitGuess::Minao,
        }
    }

    /// Run KRHF on `cell` and refresh the shared state. Returns the raw SCF
    /// result; non-convergence is recorded (see [`ScfGradScanner::converged`])
    /// but not an error — exactly as upstream's `mf_scanner(cell)` leaves the
    /// numbers on the object for the caller to inspect.
    fn run_scf(
        &self,
        cell: &Cell,
    ) -> Result<pyscf_pbc_scf::types::KScfResult, PyscfRsError> {
        let mut state = locked(&self.state, "scan")?;
        let nao = cell.mol.nao_nr;
        let nkpts = self.config.kpts.len();
        let guess = self.guess_for(&state, nao, nkpts);
        let mut mf = Krhf::new(cell.clone(), &self.config.kpts)?;
        mf.exxdiv = self.config.exxdiv;
        let result = mf.kernel(&self.config.scf_config(guess))?;
        state.last_dm = Some(result.dm.clone());
        state.last_converged = Some(result.converged);
        state.cell = Some(cell.clone());
        Ok(result)
    }

    /// `__call__(cell)` — `krhf.py:326-342`: SCF on the new cell, record the
    /// side effects, reset the (inert) grids hook, and return `(e_tot, de)`.
    ///
    /// # Errors
    /// Propagates SCF and gradient failures. A non-converged SCF is NOT an
    /// error here (see [`ScfGradScanner::converged`]); gates assert it.
    pub fn scan(&self, cell: &Cell) -> Result<(f64, Gradient), PyscfRsError> {
        self.scan_atmlst(cell, None)
    }

    /// [`ScfGradScanner::scan`] over an atom subset (`kernel(atmlst=…)`).
    ///
    /// # Errors
    /// As [`ScfGradScanner::scan`], plus an out-of-range atom id.
    pub fn scan_atmlst(
        &self,
        cell: &Cell,
        atmlst: Option<&[usize]>,
    ) -> Result<(f64, Gradient), PyscfRsError> {
        let result = self.run_scf(cell)?;
        self.reset_grids(cell);
        let mut mf = Krhf::new(cell.clone(), &self.config.kpts)?;
        mf.exxdiv = self.config.exxdiv;
        let grad = KrhfGradients::new(
            &mf,
            result.mo_energy.clone(),
            result.mo_coeff.clone(),
            result.mo_occ.clone(),
        )?;
        let grad = match atmlst {
            Some(list) => grad.with_atmlst(list.to_vec())?,
            None => grad,
        };
        let de = grad.kernel()?;
        Ok((result.e_tot, de))
    }

    /// A new [`Cell`] with `coords` swapped in (upstream's
    /// `self.cell.set_geom_(geom, inplace=False)`, `:330`), evaluated through
    /// [`ScfGradScanner::scan`].
    ///
    /// # Errors
    /// As [`ScfGradScanner::scan`], plus a coordinate-shape mismatch.
    pub fn scan_coords(
        &self,
        coords: &[[f64; 3]],
    ) -> Result<(f64, Gradient), PyscfRsError> {
        let cell = locked(&self.state, "scan_coords")?
            .cell
            .clone()
            .ok_or_else(|| invalid("KRHF scanner: no cell recorded"))?;
        let moved = crate::verify_fd::with_coords(&cell, coords)?;
        drop(cell);
        self.scan(&moved)
    }

    /// The energy half of [`ScfGradScanner::scan`]: SCF on `cell` (refreshing
    /// the shared guess state) without the gradient. This is the seam
    /// [`crate::verify_fd`] consumes.
    ///
    /// # Errors
    /// Propagates SCF failures. Non-convergence is recorded, not raised.
    pub fn energy(&self, cell: &Cell) -> Result<f64, PyscfRsError> {
        Ok(self.run_scf(cell)?.e_tot)
    }

    /// The energy half as a `Send + Sync` closure over the SHARED state, for
    /// [`crate::verify_fd`] and 18-14's optimizer.
    pub fn energy_scanner(&self) -> EnergyScanner {
        let this = self.clone();
        Box::new(move |cell: &Cell| this.energy(cell))
    }
}

/// `as_scanner(mf_grad)` — `krhf.py:300-320`.
///
/// Construction is explicit (there is no `isinstance` early return to port:
/// a value of type [`ScfGradScanner`] is already a scanner, so double-wrapping
/// cannot be expressed).
pub fn as_scanner(cell: Cell, config: KrhfScannerConfig) -> ScfGradScanner {
    ScfGradScanner::new(cell, config)
}

/// `lib.fp(g)` — `pyscf/lib/misc.py:1260-1264`: `dot(cos(arange(size)), ravel)`.
/// The ravel is C-order: element `ia * 3 + c`. Gate C compares this against
/// upstream's committed constants at upstream's own decimal count.
///
/// The products materialise into a buffer and reduce through
/// `pyscf_algebra::oracle_sum` (ALG-06: no bare `+=`).
pub fn fingerprint(gradient: &Gradient) -> f64 {
    let terms: Vec<f64> = gradient
        .iter()
        .enumerate()
        .flat_map(|(ia, row)| {
            (0..3).map(move |c| ((ia * 3 + c) as f64).cos() * row[c])
        })
        .collect();
    pyscf_algebra::oracle_sum(&terms)
}

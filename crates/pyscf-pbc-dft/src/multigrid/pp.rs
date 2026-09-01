//! `pyscf/pbc/dft/multigrid/pp.py` (256 l) — the pseudopotential on the
//! multigrid (plan 17-12, Task 4).
//!
//! # Delegated to AFTDF, not re-derived — same justified reuse v1 already
//! # ships, now stated for v2
//!
//! Upstream's `pp.py` builds `get_pp`/`get_nuc` from the SAME analytic
//! pieces `pbc.df.aft.AFTDF` and `pbc.gto.pseudo.pp_int` already use
//! (`_get_vpplocG_part1`/`get_gth_vlocG_part1`, `pp_int.get_pp_loc_part2`,
//! `ft_ao`-based `vppnl`) — the ONLY genuinely multigrid-specific piece is
//! the "pass2" step (a G-space potential contracted back to an AO matrix
//! through the collocation grid rather than through `eval_mat`), and
//! `crate::multigrid::numint`'s module doc records 17-01's own measurement
//! that this port's multigrid pass2 and FFTDF's own `eval_mat`-based pass2
//! disagree by 1e-12..1e-13 for v1 — floating-point noise, not a physical
//! effect. `crate::multigrid::pair`'s pair-fused pass2 (Task 2) has its own
//! definitional floor against v1/FFTDF (Task 5's Gate E measurement), which
//! is exactly the number this module's `get_pp`/`get_nuc` delegation
//! sidesteps re-deriving a SECOND time through a code path this plan's own
//! Task 4 instruction explicitly permits reusing (the already-shipped,
//! oracle-gated AFTDF route, 2.755e-12 — `13-VERIFICATION.md` Gate 3,
//! `crates/pyscf-pbc-df/src/gdf/nuc.rs:80`). Re-deriving `pp.py`'s full
//! `int_gauss_charge_v_rs`/`build_core_density` machinery on top of the
//! pair-fusion collocation engine, to land at the SAME already-measured
//! number, would duplicate an already-tested path for no gate this plan
//! requires — see Task 5's test for the actual comparison this delegation
//! is gated against.
//!
//! # Two SEPARATE comparisons, not one folded number
//!
//! `13-VERIFICATION.md` Gate 3 also recorded that upstream's OWN two `get_pp`
//! routes — `aft.get_pp` (AFTDF) and `pp_int.get_pp_loc_part2` (the
//! multigrid/FFTDF route) — disagree with EACH OTHER by **1.7933e-9** in
//! PySCF 2.12.1. So:
//!
//! * **this port vs its OWN AFTDF `get_nuc`/`get_pp`** (already shipped,
//!   oracle-gated at 2.755e-12) is the number [`get_nuc`]/[`get_pp`]'s own
//!   delegation trivially reproduces (it calls the SAME function) — recorded
//!   here as the "delegation is exact" statement, not a fresh measurement.
//! * **this port vs upstream's multigrid `get_pp`** (`pp.py`'s OWN route,
//!   which upstream computes DIFFERENTLY from its own AFTDF route) is
//!   Task 5's `measurements/gate_multigrid.py` `get_pp v2` number
//!   (diamond 2.411e-08, si 1.472e-07 — a MESH-INDEPENDENT floor, not a
//!   convergence residual) — reported separately, per this module's own
//!   instruction, rather than folded into one figure with the line above.
//!
//! # Scope: GAMMA POINT ONLY, no `KPoints` dependency
//!
//! Upstream's `pp.py` imports `pyscf.pbc.lib.kpts.KPoints` for the IBZ path
//! (`multigrid_pair.py:26`). `17-05` (`KPoints`) has not shipped as of this
//! plan (see the task brief's "known soft dependency gap"). This module
//! never names `KPoints` — [`get_nuc`]/[`get_pp`] take a bare [`Cell`] and
//! delegate to [`pyscf_pbc_df::aftdf`] at the gamma point only, exactly
//! mirroring `crate::multigrid::numint::MultiGridNumInt`'s already-shipped
//! v1 scope reduction. **What 17-05 needs to wire up here, once `KPoints`
//! exists**: an IBZ-restricted overload (`get_nuc_ibz`/`get_pp_ibz`, or a
//! `kpts: &KPoints` parameter added to these functions) that calls AFTDF at
//! `kpts.kpts_ibz` and unfolds with `transform_1e_operator` — the SAME
//! D-PBC-26 fast-path shape `17-07`'s `khf_ksymm::get_jk` uses, which this
//! module's plain-gamma delegation is trivially compatible with (AFTDF
//! already accepts an arbitrary k-point list; only the unfold/fold wrapper
//! is missing, and it lives in `pyscf-pbc-dft`/`pyscf-pbc-scf`, never in
//! `pyscf-pbc-df`, per D-PBC-26 rule 5).

use pyscf_pbc_df::aftdf::Aftdf;
use pyscf_pbc_gto::Cell;

use crate::error::PbcDftError;

const GAMMA: [f64; 3] = [0.0, 0.0, 0.0];

fn wrap_df(e: pyscf_pbc_df::PbcDfError) -> PbcDftError {
    PbcDftError::Core(pyscf_core::PyscfRsError::Core(
        pyscf_core::CoreError::InvalidMolecule(format!("{e}")),
    ))
}

/// `get_nuc(mydf)` on the multigrid — delegated to `AFTDF::get_nuc`, see the
/// module doc.
///
/// # Errors
/// Propagates `Aftdf::new` / `aftdf::get_nuc`.
pub fn get_nuc(cell: &Cell) -> Result<Vec<f64>, PbcDftError> {
    let df = Aftdf::new(cell.clone(), &[GAMMA]).map_err(wrap_df)?;
    let v = pyscf_pbc_df::aftdf::get_nuc(&df, &[GAMMA]).map_err(wrap_df)?;
    Ok(v[0].re.clone())
}

/// `get_pp(mydf)` on the multigrid — delegated to `AFTDF::get_pp`, see the
/// module doc.
///
/// # Errors
/// Propagates `Aftdf::new` / `aftdf::get_pp`.
pub fn get_pp(cell: &Cell) -> Result<Vec<f64>, PbcDftError> {
    let df = Aftdf::new(cell.clone(), &[GAMMA]).map_err(wrap_df)?;
    let v = pyscf_pbc_df::aftdf::get_pp(&df, &[GAMMA]).map_err(wrap_df)?;
    Ok(v[0].re.clone())
}

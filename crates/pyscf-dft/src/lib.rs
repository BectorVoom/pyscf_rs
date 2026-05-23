//! pyscf-dft: density functional theory (RKS/UKS + Becke grids + XC).
//!
//! Phase 4 (REQ-IDs: DFT-01..11). Phase 1 shipped this empty so
//! `cargo build --workspace` compiled all members (FOUND-01); plan 04-05
//! filled the XC-string parsers + XcBackend seam; plan 04-06 (this commit)
//! fills the core Kohn-Sham SCF — the `NumInt` grid loop (`nr_rks`/`nr_uks`/
//! `eval_rho`/`eval_xc`, DFT-10/D-07), the KS `get_veff = J + Vxc − hyb·K`
//! (DFT-01), the `KsOverrideHooks` trait (DFT-08), the `RKS`/`UKS` structs
//! that reuse the Phase 3 generic `kernel<H>` (DFT-01), and the D-08
//! precision seam (`DType::from_env`-driven f32/f64 dispatch through the
//! grid-loop matmul chain).
//!
//! Source: ports of `pyscf/dft/{numint,rks,uks,libxc,xcfun}.py` (Apache-2.0).
//!
//! ### Walls
//! - **pyo3-free** — the Python surface lives in pyscf-py (PyO3 wall).
//! - **cubecl-free** — all linear algebra routes through pyscf-algebra
//!   (the algebra wall, ALG-06); the grid loop carries NO `#[cube]` kernel
//!   (D-07) — AO evaluation goes through `pyscf_gto::eval_gto` (which calls
//!   `pyscf_kernels::eval_gto_sph` behind the wall) and the dense
//!   contractions are host loops with `pyscf_algebra::oracle_sum` reductions.
//! - **libxc gated** — `libxc_rs` (266 kernel crates, ~6h compile) is ONLY
//!   referenced under `--features libxc`; the default backend is xcfun_rs.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

// ── Plan 04-05 (DFT-02 parser parity + DFT-03 bit-exact eval) ──────────────
// The XC-string parsers (libxc-default + xcfun-alternate) + DftError +
// XcBackend cfg-gated seam. (04-06 layers the curated re-export surface on
// top — see the `pub use` block below — without clobbering these decls.)
pub mod error;
pub mod parser;
pub mod xc_backend;

// ── Plan 04-06 Task 1 (DFT-08/DFT-10 + D-08) ───────────────────────────────
// The core Kohn-Sham machinery: NumInt grid loop, KS get_veff, KsOverrideHooks.
pub mod hooks;
pub mod numint;
pub mod veff;
// ── Plan 04-06 Task 2 (DFT-01) ─────────────────────────────────────────────
// The RKS/UKS structs reusing the Phase 3 generic kernel<H>.
pub mod rks;
pub mod uks;

// ── Plan 04-07 Task 2 (DFT-06) ─────────────────────────────────────────────
// VV10 non-local correlation: the pure-Python _vv10nlc double-loop port +
// nr_nlc_vxc orchestration over a coarser nlcgrids (Pitfall 4, A1).
pub mod vv10;

// ── Plan 04-08 Task 1 (DFT-07) ─────────────────────────────────────────────
// DF-DFT: RKS::density_fit + DfKsHooks routing the Coulomb-J build through
// the Phase 3 pyscf-df crate (cholesky_eri/get_jk_df, D-10 reuse — no new DF
// crate); the Vxc/K parts stay on the standard grid-loop/get_jk path.
pub mod df_dft;
// ── Plan 04-08 Task 2 (DFT-07 chkfile) ─────────────────────────────────────
// KsResult chkfile persistence: impl Checkpointable for KsResult with xc/grids
// schema metadata via the Phase 3 pyscf-chkfile primitives (D-06 pattern —
// no own hdf5-metno dep, D-05 sole-owner).
pub mod chkfile;

// ── Curated re-export surface (04-06 owns this) ────────────────────────────
// XC parsing / eval (04-05 modules, surfaced here).
pub use error::DftError;
pub use parser::{Component, HybCoeffs, XcSpec};
pub use xc_backend::{DerivOrder, Family, RhoBlock, UksXcOutput, XcBackend, XcOutput};

// KS SCF surface (04-06).
pub use hooks::{KsOverrideHooks, NoKsOverrides};
pub use numint::{NrResult, NrUksResult, NumInt, XcType};
pub use rks::RKS;
pub use uks::UKS;
pub use veff::{KsVeff, default_get_veff};
// VV10 NLC surface (04-07).
pub use vv10::{NlcCoeffs, NlcResult, Vv10Output, nr_nlc_vxc, vv10nlc};
// DF-DFT surface (04-08 Task 1, DFT-07).
pub use df_dft::DfKsHooks;
// KS chkfile surface (04-08 Task 2, DFT-07 chkfile / D-06).
pub use chkfile::{GridsMeta, KsResult, dump_ks_to_file, load_ks_from_file};
// NOTE: `pub mod rks; pub mod uks;` are declared above (Task 1) so the crate
// compiles atomically; their FULL bodies (RKS/UKS reusing kernel<H>) land in
// Task 2. Task 1 ships minimal struct skeletons sufficient for the
// numint_signatures parity test; Task 2 fills the kernel reuse + oracle.

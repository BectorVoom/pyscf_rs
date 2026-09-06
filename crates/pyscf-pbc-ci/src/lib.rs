//! pyscf-pbc-ci — k-point configuration interaction.
//!
//! # `pbc/ci/cisd.py` is DEFERRED, explicitly (plan 16-13 Task 1)
//!
//! `PBC-MASTER-PLAN §8.8` pairs `KCIS` with `pbc/ci/cisd.py`. **They are
//! unrelated modules** (`16-CONTEXT §1.6`):
//!
//! * [`kcis_rhf`] is a real k-point CIS — SINGLES only, despite the phase's
//!   "CI" label — with its own Davidson (`kcis_rhf.py:97`) and a dense
//!   fallback (`:113`). **It ships.**
//! * `pbc/ci/cisd.py` (116 l) is a **Γ-point-only shim**: `RCISD.__init__` is
//!   `if abs(mf.kpt).max() > 1e-9: raise NotImplementedError` (`:24`), and the
//!   same at `:47` for `UCISD`. It subclasses the MOLECULAR `cisd.RCISD` /
//!   `ucisd.UCISD` / `gcisd` (`:18`), and **this port has no molecular CI crate
//!   at all** — there is no `crates/pyscf-ci`, and molecular CISD is not in any
//!   phase of `ROADMAP.md`.
//!
//! **Porting `pbc/ci/cisd.py` therefore means porting molecular RCISD, UCISD
//! and GCISD first, which is a phase, not a task.** It is deferred here, in
//! `16-13-SUMMARY.md` and in `ROADMAP.md`'s carry-over list — three places, so
//! it cannot be lost — rather than guessed at or silently dropped. This is
//! 17-09's discipline applied before the work instead of after it.
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::unwrap_used)]

pub mod error;
pub mod kcis_rhf;

pub use error::*;
pub use kcis_rhf::{KcisOpts, cis_diag, cis_matvec, get_kconserv_r, kernel_at_kshift, vector_size};

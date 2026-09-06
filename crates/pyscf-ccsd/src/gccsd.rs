//! `gccsd` — the NARROW molecular spin-orbital CCSD surface `pbc/cc/kccsd.py`
//! inherits from (plan 16-07 Task 1; `pyscf/cc/gccsd.py`).
//!
//! # This is a PARTIAL port, deliberately
//!
//! `pyscf/pbc/cc/kccsd.py` consumes exactly four things from the molecular
//! module, and `PBC-MASTER-PLAN §8.8` costs all of them at zero
//! (`16-CONTEXT §1.2`):
//!
//! ```text
//! pyscf/pbc/cc/kccsd.py:332   class GCCSD(gccsd.GCCSD)
//! pyscf/pbc/cc/kccsd.py:339   gccsd.GCCSD.__init__(self, mf, frozen, mo_coeff, mo_occ)
//! pyscf/pbc/cc/kccsd.py:352   gccsd.GCCSD.dump_flags(self, verbose)
//! pyscf/pbc/cc/kccsd.py:395   e_corr, self.t1, self.t2 = ccsd.CCSD.ccsd(self, t1, t2, eris)
//! pyscf/pbc/cc/kccsd.py:477   eris = gccsd._PhysicistsERIs()
//! ```
//!
//! Only `_PhysicistsERIs` carries arithmetic; the rest is object plumbing that
//! this port expresses as ordinary Rust construction and as
//! `pyscf_pbc_cc::kccsd::kernel`. **What is deliberately NOT ported**: the
//! molecular `update_amps` (`kccsd.py:68` replaces it wholesale), the molecular
//! `energy`, `make_rdm1`/`make_rdm2`, the `Lambda` equations, `spin2spatial`
//! for the molecular case, and the `_make_eris_incore`/`_make_eris_outcore`
//! AO→MO transforms, which are periodic in the k-point case and live in
//! `pyscf_pbc_cc::kccsd`. **A full molecular GCCSD belongs to whatever phase
//! actually needs molecular GHF correlation**; it is not in `ROADMAP.md` today.
//!
//! # Physicists' vs chemists' notation — the one real trap here
//!
//! `crates/pyscf-ccsd/src/eris.rs` ships [`crate::eris::ChemistsEris`], whose
//! blocks are `(pq|rs)`. `_PhysicistsERIs` holds `<pq||rs>` — the
//! ANTISYMMETRISED PHYSICIST integral, which differs from the chemist one by an
//! index transposition AND a subtraction:
//!
//! ```text
//! <pq||rs> = (pr|qs) - (ps|qr)
//! ```
//!
//! This project has already paid **+6 306 866.73 Ha** once for exactly this
//! class of misread (14-05's `decompose_j2c`, `16-CONTEXT §3.4`), so the
//! convention is in the type NAME, stated in this doc, and asserted by
//! `crates/pyscf-ccsd/tests/gccsd.rs`.

/// Marker for the antisymmetrised PHYSICIST convention `<pq||rs>`.
///
/// A zero-sized type whose only job is to make the convention impossible to
/// confuse with [`crate::eris::ChemistsEris`] at a call site. The k-point ERI
/// container that actually holds the blocks is
/// `pyscf_pbc_cc::kccsd::KgEris`, because the transform is periodic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PhysicistsEris;

impl PhysicistsEris {
    /// `<pq||rs> = (pr|qs) - (ps|qr)`, from a chemist-notation block.
    ///
    /// `chem` is `(pq|rs)` in row-major `[p][q][r][s]` order with all four
    /// dimensions equal to `n`. The output is `<pq||rs>` in the same layout.
    ///
    /// This is the scalar statement of the convention, written out so a reader
    /// can check it against the doc above without reconstructing it from a
    /// transposed array expression.
    pub fn antisymmetrise(chem: &[f64], n: usize) -> Vec<f64> {
        let at = |p: usize, q: usize, r: usize, s: usize| ((p * n + q) * n + r) * n + s;
        let mut out = vec![0.0_f64; n * n * n * n];
        for p in 0..n {
            for q in 0..n {
                for r in 0..n {
                    for s in 0..n {
                        out[at(p, q, r, s)] = chem[at(p, r, q, s)] - chem[at(p, s, q, r)];
                    }
                }
            }
        }
        out
    }
}

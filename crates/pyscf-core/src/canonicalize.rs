//! Eigenvector sign canonicalization (SCF-13, Pitfall 4 + Pitfall 12 anchor).
//!
//! Source: Inline algorithm at upstream `pyscf/scf/hf.py:1349-1357`
//! (`def eig`). Upstream does NOT expose this as a named function; Phase 3
//! extracts it here so SCF / DFT / MP2 / CCSD post-eigh paths can share
//! one impl. Pure function — no algebra dependency, no allocations.
//!
//! Algorithm: for each MO column j, find the index i_max where
//! `|c[i_max, j]|` is largest (ties broken by LOWEST index per
//! `numpy.argmax` semantics — use strict `>` comparison). If
//! `c[i_max, j]` is negative, flip the entire column. This makes
//! eigenvector signs reproducible across LAPACK vendors (MKL vs
//! OpenBLAS vs Accelerate may pick opposite signs for the same
//! eigenpair) — see RESEARCH §"Anti-Patterns" warning at line 1028.

/// In-place canonicalize signs of MO coefficient columns.
///
/// `c` is F-order: element (i, j) lives at `c[i + j*nao]`. `nao` = rows
/// (basis-function index), `nmo` = cols (MO index).
///
/// Panics in debug builds if `c.len() != nao * nmo`. In release builds the
/// loop runs over `nao * nmo` elements; over-read is a memory bug at the
/// caller — by contract callers pass a properly-sized buffer.
pub fn canonicalize_signs(c: &mut [f64], nao: usize, nmo: usize) {
    debug_assert_eq!(
        c.len(),
        nao * nmo,
        "canonicalize_signs: slice length {} != nao*nmo = {}",
        c.len(),
        nao * nmo
    );
    for j in 0..nmo {
        let col_start = j * nao;
        let mut i_max = 0usize;
        let mut abs_max = c[col_start].abs();
        for i in 1..nao {
            let v = c[col_start + i].abs();
            // STRICT GREATER-THAN: first-seen wins, matching numpy.argmax
            // tie-break to LOWEST index. RESEARCH §"Anti-Patterns" warns:
            // using `>=` here would give last-index tie-break and break
            // the Pitfall 12 cross-platform µHartree assertion.
            if v > abs_max {
                abs_max = v;
                i_max = i;
            }
        }
        if c[col_start + i_max] < 0.0 {
            for i in 0..nao {
                c[col_start + i] = -c[col_start + i];
            }
        }
    }
}

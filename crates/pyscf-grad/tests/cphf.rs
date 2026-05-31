//! The single matrix-free Krylov CPHF/CPKS solver gate (D-03, GRAD-10).
//!
//! Two always-on arms — this solver is PURE linear algebra (a matrix-free
//! response solve), so it is fully testable today against a synthetic reference
//! response operator, independent of the cintx grad-intor availability split
//! that gates the MP2 *numeric* gradient (07-01/07-03).
//!
//!   * **Krylov convergence:** for a small known linear system `(1 + A)·z = b`
//!     with `A` applied via an `aop` callback (a synthetic response operator),
//!     [`pyscf_grad::cphf::solve`] converges to the dense
//!     [`pyscf_algebra::solve_linear`] reference solution within `tol`.
//!
//!   * **Defaults:** [`pyscf_grad::cphf::solve`] honors the exact upstream
//!     defaults `max_cycle = 50`, `tol = 1e-9`, `level_shift = 0`
//!     (`cphf.py:29-31`); the caller can override `max_cycle` (the MP2 path
//!     passes 30, Pitfall 5).
//!
//!   * **`single_cphf_impl` (GRAD-10):** a structural assertion that the crate
//!     ships EXACTLY ONE `pub fn solve` CPHF implementation — `cphf::solve` —
//!     and no second copy.
//!
//! Run scoped + single-threaded (user-memory + CI discipline, NO libxc):
//!   cargo test -p pyscf-grad --locked -- --test-threads=1 cphf single_cphf_impl

use pyscf_algebra::solve_linear;
use pyscf_core::PyscfRsError;
use pyscf_grad::cphf::{self, DEFAULT_LEVEL_SHIFT, DEFAULT_MAX_CYCLE, DEFAULT_TOL};

/// A tiny synthetic CPHF problem: build a small (nvir × nocc) orbital-rotation
/// space and a dense response kernel `A` so the WHOLE solve has a closed-form
/// reference via the dense LU `solve_linear`.
///
/// The Phase-7 CPHF aop is `vind_vo(z) = fvind(z)·e_ai` (cphf.py:73-79). To make
/// the end-to-end `(1 + vind_vo)·z = b` system match a dense reference exactly,
/// we choose `mo_energy` / `mo_occ` so the (vir, occ) split + `e_ai = 1/(e_a −
/// e_i)` are known scalars per `(a, i)`, and `fvind(z) = Kmat · z` for a dense
/// `Kmat` we own. The effective dense matrix the Krylov solver inverts is then
/// `M = I + diag(e_ai) · Kmat` (since `vind_vo(z) = e_ai ⊙ (Kmat·z)`), and
/// `cphf::solve(...)` must return `z = M⁻¹ · b_base` where the solver's internal
/// RHS base is `mo1base = h1·(−e_ai)`. We therefore compare against the dense
/// solve of `M · z = mo1base`.
struct Synthetic {
    nocc: usize,
    nvir: usize,
    mo_energy: Vec<f64>,
    mo_occ: Vec<f64>,
    /// Dense response kernel, row-major (ndim × ndim), vir-major (a·nocc + i).
    kmat: Vec<f64>,
    /// e_ai = 1/(e_a − e_i), flat vir-major (a·nocc + i).
    e_ai: Vec<f64>,
}

impl Synthetic {
    fn new(nocc: usize, nvir: usize) -> Self {
        // Occupied energies below virtual energies (a real occ<vir spectrum).
        // Order the spectrum occ-first then vir so the (vir, occ) split is
        // unambiguous: mo_occ = [2,…(nocc)…, 0,…(nvir)…].
        let mut mo_energy = Vec::with_capacity(nocc + nvir);
        let mut mo_occ = Vec::with_capacity(nocc + nvir);
        for i in 0..nocc {
            mo_energy.push(-1.0 - 0.3 * i as f64);
            mo_occ.push(2.0);
        }
        for a in 0..nvir {
            mo_energy.push(0.5 + 0.4 * a as f64);
            mo_occ.push(0.0);
        }
        let e_i: Vec<f64> = mo_energy[..nocc].to_vec();
        let e_a: Vec<f64> = mo_energy[nocc..].to_vec();

        let ndim = nvir * nocc;
        let mut e_ai = vec![0.0_f64; ndim];
        for a in 0..nvir {
            for i in 0..nocc {
                e_ai[a * nocc + i] = 1.0 / (e_a[a] - e_i[i]);
            }
        }

        // A small, well-conditioned response kernel (diagonally subdominant so
        // `I + diag(e_ai)·Kmat` is comfortably non-singular and Krylov-friendly).
        let mut kmat = vec![0.0_f64; ndim * ndim];
        for r in 0..ndim {
            for c in 0..ndim {
                // Deterministic, smallish off-diagonal coupling.
                kmat[r * ndim + c] =
                    0.05 * ((r + 1) as f64) / ((c + 2) as f64) + if r == c { 0.2 } else { 0.0 };
            }
        }

        Synthetic {
            nocc,
            nvir,
            mo_energy,
            mo_occ,
            kmat,
            e_ai,
        }
    }

    fn ndim(&self) -> usize {
        self.nvir * self.nocc
    }

    /// fvind(z) = Kmat · z (dense matvec), through oracle-ordered dot.
    fn fvind(&self, z: &[f64]) -> Result<Vec<f64>, PyscfRsError> {
        let ndim = self.ndim();
        let mut out = vec![0.0_f64; ndim];
        for (r, slot) in out.iter_mut().enumerate() {
            let row = &self.kmat[r * ndim..(r + 1) * ndim];
            *slot = pyscf_algebra::oracle_dot(row, z);
        }
        Ok(out)
    }

    /// The effective dense matrix the Krylov solver inverts:
    ///   M = I + diag(e_ai) · Kmat.
    fn dense_m(&self) -> Vec<f64> {
        let ndim = self.ndim();
        let mut m = vec![0.0_f64; ndim * ndim];
        for r in 0..ndim {
            for c in 0..ndim {
                let off = self.e_ai[r] * self.kmat[r * ndim + c];
                m[r * ndim + c] = if r == c { 1.0 + off } else { off };
            }
        }
        m
    }

    /// The solver's internal RHS base: mo1base = h1 · (−e_ai).
    fn mo1base(&self, h1: &[f64]) -> Vec<f64> {
        h1.iter()
            .zip(&self.e_ai)
            .map(|(&hi, &eai)| hi * -eai)
            .collect()
    }
}

#[test]
fn cphf_krylov_converges_to_dense_reference() {
    let s = Synthetic::new(2, 3);
    let ndim = s.ndim();

    // An arbitrary RHS h1 (flat vir-major).
    let h1: Vec<f64> = (0..ndim).map(|k| 0.1 + 0.07 * k as f64).collect();

    // Analytical reference: solve M · z = mo1base densely (algebra-wall LU).
    let m = s.dense_m();
    let rhs = s.mo1base(&h1);
    let z_ref = solve_linear(&m, &rhs, ndim).expect("dense reference solve");

    // The matrix-free Krylov solve. fvind closes over the synthetic kernel.
    let fvind = |z: &[f64]| s.fvind(z);
    let z = cphf::solve(
        &fvind,
        &s.mo_energy,
        &s.mo_occ,
        &h1,
        None, // s1 = None → the Phase-7 solve_nos1 path.
        DEFAULT_MAX_CYCLE,
        DEFAULT_TOL,
        false, // hermi
        DEFAULT_LEVEL_SHIFT,
    )
    .expect("cphf::solve converges");

    assert_eq!(z.len(), ndim, "solution length must be nvir·nocc");
    let max_diff = z
        .iter()
        .zip(&z_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-8,
        "Krylov solution disagrees with dense reference: max|Δ| = {max_diff:e} (z={z:?}, ref={z_ref:?})"
    );

    // Sanity: verify the converged z actually satisfies (1 + vind_vo)·z = mo1base.
    let kz = s.fvind(&z).expect("fvind");
    let mut residual = 0.0_f64;
    for k in 0..ndim {
        let lhs = z[k] + s.e_ai[k] * kz[k]; // (I + diag(e_ai)·Kmat)·z
        residual = residual.max((lhs - rhs[k]).abs());
    }
    assert!(
        residual < 1e-7,
        "converged z does not satisfy the response equation: residual = {residual:e}"
    );
}

#[test]
fn cphf_defaults_are_exact_upstream() {
    // The locked upstream defaults (cphf.py:29-31): max_cycle=50, tol=1e-9,
    // level_shift=0.
    assert_eq!(
        DEFAULT_MAX_CYCLE, 50,
        "upstream cphf.py:30 default max_cycle"
    );
    assert_eq!(DEFAULT_TOL, 1e-9, "upstream cphf.py:30 default tol");
    assert_eq!(
        DEFAULT_LEVEL_SHIFT, 0.0,
        "upstream cphf.py:31 default level_shift"
    );
}

#[test]
fn cphf_caller_can_override_max_cycle() {
    // The MP2 caller overrides max_cycle=30 (Pitfall 5 / mp2.py:280). A well-
    // conditioned tiny system converges well within 30 cycles, so passing 30
    // (NOT the 50 default) must still converge to the dense reference.
    let s = Synthetic::new(1, 2);
    let ndim = s.ndim();
    let h1: Vec<f64> = (0..ndim).map(|k| 0.2 - 0.05 * k as f64).collect();

    let m = s.dense_m();
    let rhs = s.mo1base(&h1);
    let z_ref = solve_linear(&m, &rhs, ndim).expect("dense reference");

    let fvind = |z: &[f64]| s.fvind(z);
    let z = cphf::solve(
        &fvind,
        &s.mo_energy,
        &s.mo_occ,
        &h1,
        None,
        30, // ← MP2 override, NOT the 50 default.
        DEFAULT_TOL,
        false,
        DEFAULT_LEVEL_SHIFT,
    )
    .expect("cphf::solve converges with max_cycle=30");

    let max_diff = z
        .iter()
        .zip(&z_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-8,
        "max_cycle=30 override failed to converge: max|Δ| = {max_diff:e}"
    );
}

#[test]
fn cphf_withs1_branch_is_rejected_this_phase() {
    // solve_withs1 (field-dependent, s1 != None) is the Hessian future, NOT
    // Phase 7. Passing s1=Some must return a clear error, never a wrong tensor.
    let s = Synthetic::new(1, 1);
    let ndim = s.ndim();
    let h1 = vec![0.1_f64; ndim];
    let s1 = vec![0.0_f64; ndim];
    let fvind = |z: &[f64]| s.fvind(z);
    let r = cphf::solve(
        &fvind,
        &s.mo_energy,
        &s.mo_occ,
        &h1,
        Some(&s1),
        DEFAULT_MAX_CYCLE,
        DEFAULT_TOL,
        false,
        DEFAULT_LEVEL_SHIFT,
    );
    assert!(
        r.is_err(),
        "s1=Some (solve_withs1) must be rejected this phase"
    );
}

#[test]
fn cphf_trivial_rhs_returns_zero() {
    // A zero RHS has the trivial solution z = 0 (lib.krylov lines 1293-1295);
    // the solver must short-circuit rather than iterate.
    let s = Synthetic::new(2, 2);
    let ndim = s.ndim();
    let h1 = vec![0.0_f64; ndim];
    let fvind = |z: &[f64]| s.fvind(z);
    let z = cphf::solve(
        &fvind,
        &s.mo_energy,
        &s.mo_occ,
        &h1,
        None,
        DEFAULT_MAX_CYCLE,
        DEFAULT_TOL,
        false,
        DEFAULT_LEVEL_SHIFT,
    )
    .expect("trivial solve");
    assert_eq!(z.len(), ndim);
    assert!(z.iter().all(|&v| v == 0.0), "zero RHS ⇒ zero solution");
}

#[test]
fn cphf_aop_error_propagates() {
    // A fvind that returns an Err (e.g. a missing-intor cintx-availability
    // error) must `?`-propagate straight out of the solve — never a panic, never
    // a wrong tensor.
    let s = Synthetic::new(1, 2);
    let ndim = s.ndim();
    let h1 = vec![0.3_f64; ndim];
    let fvind = |_z: &[f64]| -> Result<Vec<f64>, PyscfRsError> {
        Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            "synthetic cintx-availability error from fvind".into(),
        )))
    };
    let r = cphf::solve(
        &fvind,
        &s.mo_energy,
        &s.mo_occ,
        &h1,
        None,
        DEFAULT_MAX_CYCLE,
        DEFAULT_TOL,
        false,
        DEFAULT_LEVEL_SHIFT,
    );
    assert!(r.is_err(), "fvind error must propagate out of cphf::solve");
}

/// GRAD-10 — the single-CPHF-implementation structural assertion (D-03).
///
/// There must be EXACTLY ONE CPHF solver in `pyscf-grad`: `cphf::solve`. This
/// guards against a second `pub fn solve` CPHF copy landing in another module
/// (e.g. a per-method CPHF in mp2.rs / ccsd.rs) — the GRAD-10 violation. We scan
/// the crate's own `src/` for `pub fn solve(` declarations that look like a CPHF
/// solver and assert there is exactly one, located in `cphf.rs`.
#[test]
fn single_cphf_impl() {
    use std::fs;
    use std::path::Path;

    let src_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    assert!(src_dir.is_dir(), "src/ dir must exist: {src_dir:?}");

    let mut cphf_solve_sites: Vec<String> = Vec::new();
    for entry in fs::read_dir(&src_dir).expect("read src/") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("rs") {
            continue;
        }
        let text = fs::read_to_string(&path).expect("read source file");
        let fname = path
            .file_name()
            .and_then(|f| f.to_str())
            .unwrap_or("")
            .to_string();
        for line in text.lines() {
            let t = line.trim_start();
            // A CPHF solver entry point: `pub fn solve(` (the cphf.solve port).
            // Restrict to a `pub fn solve(` declaration so we don't trip on
            // `cphf::solve` *call sites* (those are `cphf::solve(` with a path
            // prefix and never start the trimmed line with `pub fn`).
            if t.starts_with("pub fn solve(") {
                cphf_solve_sites.push(format!("{fname}: {}", t.trim()));
            }
        }
    }

    assert_eq!(
        cphf_solve_sites.len(),
        1,
        "GRAD-10: there must be EXACTLY ONE CPHF `pub fn solve(` in pyscf-grad \
         (D-03 — one solver, not N). Found: {cphf_solve_sites:?}"
    );
    assert!(
        cphf_solve_sites[0].starts_with("cphf.rs:"),
        "GRAD-10: the single CPHF solver must live in cphf.rs, found in {:?}",
        cphf_solve_sites[0]
    );
}

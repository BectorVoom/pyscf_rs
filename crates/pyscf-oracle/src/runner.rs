//! Oracle runner — dispatches methods to per-check helpers, drives Python via
//! `Python::attach` (pyo3 in dev-deps only — ORACLE-01).
//!
//! All 8 SCF/DF arms ship with real implementations (checker iteration 1
//! BLOCKER 1 closure). ROADMAP success criterion 6: every SCF success
//! criterion is asserted via this macro. Phase 4 plan 04-04 adds a 9th arm,
//! `grid_weights` (DFT-04/09 byte-for-byte Becke grid compare).
//!
//! Build modes:
//!   - default features (no `python`): only `OracleError` + the dispatch
//!     skeleton compile; `run_oracle_check` returns
//!     `OracleError::PythonFeatureNotEnabled` for any method except
//!     `"<UNKNOWN_BUT_KNOWN>"`-style dispatch errors (which surface as
//!     `OracleError::UnknownMethod` before any Python work).
//!   - `--features python`: full Python::attach + upstream pyscf driver
//!     for each of the 8 arms.
//!
//! This split lets `cargo build -p pyscf-oracle` succeed in environments
//! without libpython.so while CI runners opt into the live oracle by
//! adding `--features python`.

use thiserror::Error;

#[derive(Debug, Error)]
pub enum OracleError {
    #[error("oracle: pyscf-rs returned error: {0}")]
    PyscfRs(String),
    #[error("oracle: upstream PySCF returned error: {0}")]
    Upstream(String),
    #[error("oracle: |pyscf-rs - upstream| = {diff:e} > tol={tol:e} for {key}")]
    Diff { diff: f64, tol: f64, key: String },
    #[error("oracle: unknown method '{0}'")]
    UnknownMethod(String),
    #[error(
        "oracle: pyscf-oracle was built without the `python` feature \
         — rebuild with `cargo test --features python` to drive upstream pyscf"
    )]
    PythonFeatureNotEnabled,
    #[error("oracle: tempfile error: {0}")]
    Io(#[from] std::io::Error),
}

/// All known method names — recognised by dispatch regardless of whether
/// the `python` feature is enabled. Used to distinguish a genuinely
/// unknown method (always an error) from a known method that can't run
/// because Python isn't compiled in. (8 SCF/DF arms + `grid_weights`.)
const KNOWN_METHODS: &[&str] = &[
    "scf_rhf_energy",
    "scf_uhf_energy",
    "scf_diis_iter_count",
    "scf_init_guess",
    "df_hf_energy",
    "chkfile_roundtrip",
    "mulliken_pop",
    "dip_moment",
    // DFT-04 / DFT-09 (plan 04-04): byte-for-byte Becke grid coords + weights
    // vs upstream `dft.gen_grid.Grids`. The fixture name encodes the level as
    // `<base>@levelN` (e.g. "h2o_ccpvdz@level3").
    "grid_weights",
    // DFT-01 (plan 04-06): RKS / UKS total energy ≤ 1 µHartree vs upstream
    // `dft.RKS(mol, xc=...).kernel()` / `dft.UKS(...)`. The fixture name
    // encodes the XC functional as `<base>@<xc>` (e.g. "h2o_ccpvdz@svwn",
    // "h2o_ccpvdz@b3lyp"). f64 ONLY — these are the bit-exact oracle path
    // (PYSCF_DTYPE unset); the f32 path is NEVER compared to upstream (D-08).
    "rks_energy",
    "uks_energy",
    // DFT-07 (plan 04-08): DF-DFT total energy ≤ 1 µHartree vs upstream
    // `dft.RKS(mol, xc=...).density_fit().kernel()`. The Coulomb-J build routes
    // through the Phase 3 pyscf-df crate (D-10 reuse); the fixture encodes the
    // XC functional as `<base>@<xc>` (e.g. "h2o_ccpvdz@svwn"). f64 ONLY.
    "df_dft_energy",
    // DFT-07 chkfile (plan 04-08): KsResult dump → h5py-readable on the upstream
    // /scf schema + the xc/grids metadata, round-tripping both directions (the
    // ORACLE-08 harness extended to the KS result type, D-06).
    "ks_chkfile_roundtrip",
    // ── Phase 5 MP2 numeric oracle arms (plan 05-01) ───────────────────────
    // MP2-01/02/04/05: RMP2 / UMP2 / DF-MP2 (conventional + native) correlation
    // energies + MP2 RDM, each compared ≤ 1 µHartree vs upstream
    // `mp.RMP2(mf).kernel()` / `mp.UMP2(uhf_mf).kernel()` /
    // `mp.DFMP2(mf).kernel()` (and the native RI-MP2 path). The fixture name
    // encodes the system as `<base>` (e.g. "h2o_ccpvdz"), exactly the
    // rks_energy precedent. ALL five MP2 numeric arms are CI-gated behind
    // cintx#11 (the numeric oracle needs both arity-4 `int2e` for the in-core
    // RMP2/UMP2 path AND arity-3 `int3c2e_sph` for the DF-MP2 path); the
    // `mp2-oracle-cintx-gated` CI job stays `if: false` until cintx#11 ships
    // these — mirroring the DF-HF / DFT-01 (rks_energy/uks_energy) gating.
    "mp2_rmp2_energy",
    "mp2_ump2_energy",
    "dfmp2_energy",
    "dfmp2_native_energy",
    "mp2_rdm",
    // ── Phase 6 CCSD numeric oracle arms (plan 06-11) ──────────────────────
    // CCSD-01/02/05/06/08: RCCSD / UCCSD / DF-CCSD correlation energies + λ
    // amplitudes + 1-/2-particle RDMs, each compared byte-identity vs upstream
    // `cc.CCSD(mf).kernel()` / `cc.UCCSD(uhf_mf).kernel()` /
    // `cc.CCSD(mf.density_fit()).kernel()` (and `.solve_lambda()` /
    // `.make_rdm1()` / `.make_rdm2()`). The fixture name encodes the system as
    // `<base>` (e.g. "h2o_ccpvdz", "caffeine_ccpvdz"), exactly the
    // mp2_rmp2_energy precedent.
    //
    // ALL SIX are registered here so the always-on dispatch layer recognises
    // them (the `unknown method` guard distinguishes a genuinely unknown name
    // from a known-but-Python-gated arm). The live byte-identity comparison
    // itself is `workflow_dispatch`/human-verify ONLY (the
    // `ccsd-oracle-upstream-manual` CI arm — caffeine/cc-pVDZ byte-identity,
    // DF-CCSD spill, λ/RDM byte-identity) because it needs an installed upstream
    // PySCF. The always-on, in-tree small-system numeric proofs live in
    // crates/pyscf-ccsd/tests/ (rccsd_numeric_smoke, uccsd_smoke, lambda, rdm,
    // dfccsd_spill, direct) — never pulling libxc, never calling live PySCF
    // (user-memory constraint; the 05-08 mp2-oracle-upstream-manual precedent).
    "ccsd_rccsd_energy",
    "ccsd_uccsd_energy",
    "ccsd_dfccsd_energy",
    "ccsd_lambda",
    "ccsd_rdm1",
    "ccsd_rdm2",
    // ── Phase 7 gradient + geomopt oracle arms (plan 07-10) ────────────────
    // GRAD-01..07 + GEOMOPT-07: per-method analytical nuclear gradients
    // (`nuc_grad_*`) + the H2O geometry-optimizer trajectory fixture
    // (`geomopt_h2o`), registered here so the always-on dispatch layer
    // RECOGNISES them (the `unknown method` guard distinguishes a genuinely
    // unknown name from a known-but-Python-gated arm), mirroring the MP2/CCSD
    // register-but-defer-dispatch precedent.
    //
    // REGISTER-BUT-DEFER-DISPATCH: the names are registered now; the live
    // byte-identity comparison (≤ 1e-7 Ha/Bohr vs `pyscf/grad/*`) + the
    // geomopt trajectory/stationary-point parity vs `geometric_solver` are
    // `workflow_dispatch` / human-verify ONLY (the `grad-oracle-upstream-manual`
    // CI arm), because they need an installed upstream PySCF + geomeTRIC AND
    // they ride the cintx grad-intor gate: SIX of the eight gradient-integral
    // families (`int2e_ip1`, `int1e_ip{ovlp,kin,nuc,rinv}`, `ECPscalar_iprinv`
    // + the `with_rinv_at_nucleus` origin shift) are MISSING from cintx today
    // (07-01-SUMMARY), with no scheduled cintx workstream — so the numeric arms
    // stay gated until that workstream lands. Only `int3c2e_ip1` (DF-grad) and
    // `int1e_ecp_ipnuc` (= `ECPscalar_ipnuc`, the ECP `get_hcore` term) are
    // cintx-ready.
    //
    // The ALWAYS-ON numeric proofs are the in-tree FD self-verification
    // (`verify_fd`, D-01, ≤ 1e-6 Ha/Bohr — NO upstream PySCF) + the structural
    // arms (`single_cphf_impl`, `atmlst`, optimizer convergence) living in
    // crates/pyscf-grad/tests + crates/pyscf-geomopt/tests (run by the
    // always-on `grad-structural` CI arm). The DEFAULT `pyscf-oracle` build
    // pulls NO libxc and makes NO live PySCF call (ORACLE_NO_LIBXC discipline,
    // 06-11).
    "nuc_grad_rhf",
    "nuc_grad_uhf",
    "nuc_grad_rks",
    "nuc_grad_uks",
    "nuc_grad_mp2",
    "nuc_grad_ccsd",
    "nuc_grad_ecp",
    "geomopt_h2o",
];

/// Top-level dispatcher. Resolves the method name to either a known
/// implementation or an `UnknownMethod` error. Without the `python`
/// feature, known methods short-circuit to `PythonFeatureNotEnabled`.
pub fn run_oracle_check(
    method: &str,
    fixture_name: &str,
    tolerance: f64,
) -> Result<(), OracleError> {
    if !KNOWN_METHODS.contains(&method) {
        return Err(OracleError::UnknownMethod(method.to_string()));
    }
    #[cfg(feature = "python")]
    {
        return python_impl::dispatch(method, fixture_name, tolerance);
    }
    #[cfg(not(feature = "python"))]
    {
        // Hush unused-arg warnings in the no-feature build.
        let _ = (fixture_name, tolerance);
        Err(OracleError::PythonFeatureNotEnabled)
    }
}

// ─────────────────────────────────────────────────────────────────────────
// Python-feature half — all 8 arms.
// ─────────────────────────────────────────────────────────────────────────

#[cfg(feature = "python")]
mod python_impl {
    use super::OracleError;
    use crate::fixtures;
    use pyo3::prelude::*;
    use pyo3::types::PyAnyMethods;

    pub(super) fn dispatch(method: &str, fixture: &str, tol: f64) -> Result<(), OracleError> {
        Python::attach(|py| match method {
            "scf_rhf_energy" => check_scf_rhf_energy(py, fixture, tol),
            "scf_uhf_energy" => check_scf_uhf_energy(py, fixture, tol),
            "scf_diis_iter_count" => check_diis_iter_count(py, fixture, tol),
            "scf_init_guess" => check_init_guess(py, fixture, tol),
            "df_hf_energy" => check_df_hf_energy(py, fixture, tol),
            "chkfile_roundtrip" => check_chkfile_roundtrip(py, fixture, tol),
            "mulliken_pop" => check_mulliken_pop(py, fixture, tol),
            "dip_moment" => check_dip_moment(py, fixture, tol),
            "grid_weights" => check_grid_weights(py, fixture, tol),
            "rks_energy" => check_rks_energy(py, fixture, tol),
            "uks_energy" => check_uks_energy(py, fixture, tol),
            "df_dft_energy" => check_df_dft_energy(py, fixture, tol),
            "ks_chkfile_roundtrip" => check_ks_chkfile_roundtrip(py, fixture, tol),
            other => Err(OracleError::UnknownMethod(other.to_string())),
        })
    }

    // ── helpers ──────────────────────────────────────────────────────────

    fn build_pyscf_mol<'py>(
        py: Python<'py>,
        fixture: &str,
    ) -> Result<Bound<'py, PyAny>, OracleError> {
        // `pyscf.gto.M` (alias `pyscf.M`) constructs + builds a Mole in one
        // call given kwargs. We use it via the top-level `pyscf.M` symbol.
        let pyscf = py
            .import("pyscf")
            .map_err(|e| OracleError::Upstream(format!("import pyscf: {}", e)))?;
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs
            .set_item("atom", fixtures::atom(fixture))
            .map_err(|e| OracleError::Upstream(format!("kwargs atom: {}", e)))?;
        kwargs
            .set_item("basis", fixtures::basis(fixture))
            .map_err(|e| OracleError::Upstream(format!("kwargs basis: {}", e)))?;
        let s = fixtures::spin(fixture);
        if s != 0 {
            kwargs
                .set_item("spin", s)
                .map_err(|e| OracleError::Upstream(format!("kwargs spin: {}", e)))?;
        }
        let mol = pyscf
            .call_method("M", (), Some(&kwargs))
            .map_err(|e| OracleError::Upstream(format!("M(): {}", e)))?;
        Ok(mol)
    }

    fn build_rust_mol(fixture: &str) -> Result<pyscf_core::Mole, OracleError> {
        let atom = fixtures::atom(fixture);
        let basis = fixtures::basis(fixture);
        let spin = fixtures::spin(fixture);
        let args = pyscf_gto::MoleBuildArgs {
            atom: pyscf_gto::AtomInput::String(atom.into()),
            basis: pyscf_gto::BasisInput::Name(basis.into()),
            spin,
            ..Default::default()
        };
        pyscf_gto::M(args).map_err(|e| OracleError::PyscfRs(format!("M: {}", e)))
    }

    /// Extract a Vec<f64> from a numpy array via `.flatten().tolist()`.
    /// Avoids a hard numpy crate dep — the oracle path only needs a
    /// best-effort numeric comparison, not zero-copy.
    fn np_to_vec_f64(arr: &Bound<'_, PyAny>) -> Result<Vec<f64>, OracleError> {
        arr.call_method0("flatten")
            .map_err(|e| OracleError::Upstream(format!("flatten: {}", e)))?
            .call_method0("tolist")
            .map_err(|e| OracleError::Upstream(format!("tolist: {}", e)))?
            .extract()
            .map_err(|e| OracleError::Upstream(format!("extract Vec<f64>: {}", e)))
    }

    fn elementwise_max_diff(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    // ── Arm 1: RHF energy ────────────────────────────────────────────────

    fn check_scf_rhf_energy(py: Python<'_>, fixture: &str, tol: f64) -> Result<(), OracleError> {
        let mol = build_pyscf_mol(py, fixture)?;
        let scf = py
            .import("pyscf.scf")
            .map_err(|e| OracleError::Upstream(format!("import scf: {}", e)))?;
        let mf = scf
            .call_method1("RHF", (mol,))
            .map_err(|e| OracleError::Upstream(format!("RHF(): {}", e)))?;
        let e_upstream: f64 = mf
            .call_method0("kernel")
            .map_err(|e| OracleError::Upstream(format!("kernel: {}", e)))?
            .extract()
            .map_err(|e| OracleError::Upstream(format!("extract f64: {}", e)))?;

        let rust_mol = build_rust_mol(fixture)?;
        let mut rhf = pyscf_scf::RHF::new(rust_mol);
        rhf.kernel()
            .map_err(|e| OracleError::PyscfRs(format!("kernel: {}", e)))?;
        let e_rs = rhf.e_tot;

        let diff = (e_rs - e_upstream).abs();
        if diff > tol {
            return Err(OracleError::Diff {
                diff,
                tol,
                key: format!("scf_rhf_energy[{}]", fixture),
            });
        }
        Ok(())
    }

    // ── Arm 2: UHF energy ────────────────────────────────────────────────

    fn check_scf_uhf_energy(py: Python<'_>, fixture: &str, tol: f64) -> Result<(), OracleError> {
        let mol = build_pyscf_mol(py, fixture)?;
        let scf = py
            .import("pyscf.scf")
            .map_err(|e| OracleError::Upstream(format!("import scf: {}", e)))?;
        let mf = scf
            .call_method1("UHF", (mol,))
            .map_err(|e| OracleError::Upstream(format!("UHF(): {}", e)))?;
        let e_upstream: f64 = mf
            .call_method0("kernel")
            .map_err(|e| OracleError::Upstream(format!("kernel: {}", e)))?
            .extract()
            .map_err(|e| OracleError::Upstream(format!("extract f64: {}", e)))?;

        let rust_mol = build_rust_mol(fixture)?;
        let mut uhf = pyscf_scf::UHF::new(rust_mol);
        uhf.kernel()
            .map_err(|e| OracleError::PyscfRs(format!("UHF kernel: {}", e)))?;
        let e_rs = uhf.e_tot;

        let diff = (e_rs - e_upstream).abs();
        if diff > tol {
            return Err(OracleError::Diff {
                diff,
                tol,
                key: format!("scf_uhf_energy[{}]", fixture),
            });
        }
        Ok(())
    }

    // ── Arm 3: DIIS iteration count (|Δcycles| ≤ tol; integer-ish tol) ───

    fn check_diis_iter_count(py: Python<'_>, fixture: &str, tol: f64) -> Result<(), OracleError> {
        let mol = build_pyscf_mol(py, fixture)?;
        let scf = py
            .import("pyscf.scf")
            .map_err(|e| OracleError::Upstream(format!("import scf: {}", e)))?;
        let mf = scf
            .call_method1("RHF", (mol,))
            .map_err(|e| OracleError::Upstream(format!("RHF(): {}", e)))?;
        mf.call_method0("kernel")
            .map_err(|e| OracleError::Upstream(format!("kernel: {}", e)))?;
        // pyscf records the converged-cycle count in `.cycles` (attr added
        // by the SCF base in pyscf/scf/hf.py).
        let cycles_upstream: i64 = mf
            .getattr("cycles")
            .map_err(|e| OracleError::Upstream(format!("cycles attr: {}", e)))?
            .extract()
            .map_err(|e| OracleError::Upstream(format!("extract cycles: {}", e)))?;

        let rust_mol = build_rust_mol(fixture)?;
        let mut rhf = pyscf_scf::RHF::new(rust_mol);
        rhf.kernel()
            .map_err(|e| OracleError::PyscfRs(format!("kernel: {}", e)))?;
        let cycles_rs = rhf.cycles as i64;

        let diff = (cycles_rs - cycles_upstream).unsigned_abs() as f64;
        if diff > tol {
            return Err(OracleError::Diff {
                diff,
                tol,
                key: format!(
                    "scf_diis_iter_count[{}] (rs={}, upstream={})",
                    fixture, cycles_rs, cycles_upstream
                ),
            });
        }
        Ok(())
    }

    // ── Arm 4: init_guess first-iteration density ────────────────────────

    fn check_init_guess(py: Python<'_>, fixture: &str, tol: f64) -> Result<(), OracleError> {
        // Fixture is "<base>_<mode>" e.g. "h2o_ccpvdz_minao". The base
        // (without the `_<mode>` suffix) is what we hand both sides.
        let mode = fixtures::init_guess_mode(fixture);
        // Strip the mode suffix so atom/basis lookup resolves the base
        // fixture; if no suffix is present, fixtures::atom() still works.
        let base_owned = fixture.trim_end_matches(&format!("_{}", mode)).to_string();
        let base = base_owned.as_str();

        let mol = build_pyscf_mol(py, base)?;
        let scf = py
            .import("pyscf.scf")
            .map_err(|e| OracleError::Upstream(format!("import scf: {}", e)))?;
        let mf = scf
            .call_method1("RHF", (mol,))
            .map_err(|e| OracleError::Upstream(format!("RHF(): {}", e)))?;
        // mf.get_init_guess(key=mode) — pyscf's positional API is
        // get_init_guess(mol=None, key='minao').
        let dm0_upstream = mf
            .call_method1("get_init_guess", (py.None(), mode))
            .map_err(|e| OracleError::Upstream(format!("get_init_guess[{}]: {}", mode, e)))?;
        let dm0_upstream_vec = np_to_vec_f64(&dm0_upstream)?;

        let rust_mol = build_rust_mol(base)?;
        let im = pyscf_scf::parse_init_guess_mode(mode)
            .map_err(|e| OracleError::PyscfRs(format!("parse init_guess mode: {}", e)))?;
        let dm0_rs = pyscf_scf::default_get_init_guess(&rust_mol, &im)
            .map_err(|e| OracleError::PyscfRs(format!("get_init_guess: {}", e)))?;

        if dm0_rs.data.len() != dm0_upstream_vec.len() {
            return Err(OracleError::Diff {
                diff: f64::INFINITY,
                tol,
                key: format!(
                    "init_guess[{},{}] shape mismatch (rs={}, upstream={})",
                    fixture,
                    mode,
                    dm0_rs.data.len(),
                    dm0_upstream_vec.len()
                ),
            });
        }
        let max_diff = elementwise_max_diff(&dm0_rs.data, &dm0_upstream_vec);
        if max_diff > tol {
            return Err(OracleError::Diff {
                diff: max_diff,
                tol,
                key: format!("init_guess[{},{}] element-wise max", fixture, mode),
            });
        }
        Ok(())
    }

    // ── Arm 5: DF-HF energy ──────────────────────────────────────────────

    fn check_df_hf_energy(py: Python<'_>, fixture: &str, tol: f64) -> Result<(), OracleError> {
        let mol = build_pyscf_mol(py, fixture)?;
        let scf = py
            .import("pyscf.scf")
            .map_err(|e| OracleError::Upstream(format!("import scf: {}", e)))?;
        let mf = scf
            .call_method1("RHF", (mol,))
            .map_err(|e| OracleError::Upstream(format!("RHF(): {}", e)))?;
        let mf_df = mf
            .call_method0("density_fit")
            .map_err(|e| OracleError::Upstream(format!("density_fit: {}", e)))?;
        let e_upstream: f64 = mf_df
            .call_method0("kernel")
            .map_err(|e| OracleError::Upstream(format!("kernel: {}", e)))?
            .extract()
            .map_err(|e| OracleError::Upstream(format!("extract: {}", e)))?;

        let rust_mol = build_rust_mol(fixture)?;
        let rhf = pyscf_scf::RHF::new(rust_mol)
            .density_fit(None)
            .map_err(|e| OracleError::PyscfRs(format!("density_fit: {}", e)))?;
        // density_fit(None) consumes self and returns the configured RHF.
        // We still need a mutable handle to call kernel().
        let mut rhf = rhf;
        rhf.kernel()
            .map_err(|e| OracleError::PyscfRs(format!("DF kernel: {}", e)))?;
        let e_rs = rhf.e_tot;

        let diff = (e_rs - e_upstream).abs();
        if diff > tol {
            return Err(OracleError::Diff {
                diff,
                tol,
                key: format!("df_hf_energy[{}]", fixture),
            });
        }
        Ok(())
    }

    // ── Arm 6: ORACLE-08 chkfile round-trip (both directions) ────────────

    fn check_chkfile_roundtrip(py: Python<'_>, fixture: &str, tol: f64) -> Result<(), OracleError> {
        // Direction (a): PySCF writes → pyscf-rs reads.
        let tmp_a = tempfile::NamedTempFile::new()?;
        let path_a = tmp_a.path();
        {
            let mol = build_pyscf_mol(py, fixture)?;
            let scf = py
                .import("pyscf.scf")
                .map_err(|e| OracleError::Upstream(format!("import scf: {}", e)))?;
            let mf = scf
                .call_method1("RHF", (mol,))
                .map_err(|e| OracleError::Upstream(format!("RHF(): {}", e)))?;
            mf.setattr(
                "chkfile",
                path_a
                    .to_str()
                    .ok_or_else(|| OracleError::Upstream("path_a not utf8".into()))?,
            )
            .map_err(|e| OracleError::Upstream(format!("setattr chkfile: {}", e)))?;
            mf.call_method0("kernel")
                .map_err(|e| OracleError::Upstream(format!("kernel: {}", e)))?;
        }
        // pyscf-rs reads the chkfile that PySCF just wrote.
        let result = pyscf_scf::load_scf_from_file(path_a)
            .map_err(|e| OracleError::PyscfRs(format!("load_scf_from_file: {}", e)))?;
        // Cross-check e_tot against upstream's own re-read of the chkfile.
        let pyscf_lib = py
            .import("pyscf.lib.chkfile")
            .map_err(|e| OracleError::Upstream(format!("import pyscf.lib.chkfile: {}", e)))?;
        let chk_data = pyscf_lib
            .call_method1(
                "load",
                (
                    path_a
                        .to_str()
                        .ok_or_else(|| OracleError::Upstream("path_a not utf8".into()))?,
                    "scf",
                ),
            )
            .map_err(|e| OracleError::Upstream(format!("chkfile.load: {}", e)))?;
        let e_upstream: f64 = chk_data
            .get_item("e_tot")
            .map_err(|e| OracleError::Upstream(format!("get e_tot: {}", e)))?
            .extract()
            .map_err(|e| OracleError::Upstream(format!("extract e_tot: {}", e)))?;
        let diff_a = (result.e_tot.0 - e_upstream).abs();
        if diff_a > tol {
            return Err(OracleError::Diff {
                diff: diff_a,
                tol,
                key: format!("chkfile_roundtrip[A,{}]", fixture),
            });
        }

        // Direction (b): pyscf-rs writes → PySCF from_chk reads + reruns.
        let tmp_b = tempfile::NamedTempFile::new()?;
        let path_b = tmp_b.path();
        let rust_mol = build_rust_mol(fixture)?;
        let mut rhf = pyscf_scf::RHF::new(rust_mol);
        rhf.chkfile = Some(path_b.to_path_buf());
        rhf.kernel()
            .map_err(|e| OracleError::PyscfRs(format!("kernel: {}", e)))?;

        // Upstream reads our chkfile via from_chk() then reruns kernel().
        // If hdf5-metno's payload is bit-compatible with h5py, the rerun
        // converges to the pyscf-rs energy within tolerance.
        let mol = build_pyscf_mol(py, fixture)?;
        let scf = py
            .import("pyscf.scf")
            .map_err(|e| OracleError::Upstream(format!("import scf: {}", e)))?;
        let mf = scf
            .call_method1("RHF", (mol,))
            .map_err(|e| OracleError::Upstream(format!("RHF(): {}", e)))?;
        mf.call_method1(
            "from_chk",
            (path_b
                .to_str()
                .ok_or_else(|| OracleError::Upstream("path_b not utf8".into()))?,),
        )
        .map_err(|e| OracleError::Upstream(format!("from_chk: {}", e)))?;
        let e_upstream_b: f64 = mf
            .call_method0("kernel")
            .map_err(|e| OracleError::Upstream(format!("kernel after from_chk: {}", e)))?
            .extract()
            .map_err(|e| OracleError::Upstream(format!("extract: {}", e)))?;
        let diff_b = (rhf.e_tot - e_upstream_b).abs();
        if diff_b > tol {
            return Err(OracleError::Diff {
                diff: diff_b,
                tol,
                key: format!("chkfile_roundtrip[B,{}]", fixture),
            });
        }
        Ok(())
    }

    // ── Arm 7: Mulliken population (atom-resolved charges) ───────────────

    fn check_mulliken_pop(py: Python<'_>, fixture: &str, tol: f64) -> Result<(), OracleError> {
        let mol = build_pyscf_mol(py, fixture)?;
        let scf = py
            .import("pyscf.scf")
            .map_err(|e| OracleError::Upstream(format!("import scf: {}", e)))?;
        let mf = scf
            .call_method1("RHF", (mol,))
            .map_err(|e| OracleError::Upstream(format!("RHF(): {}", e)))?;
        mf.call_method0("kernel")
            .map_err(|e| OracleError::Upstream(format!("kernel: {}", e)))?;
        // pop, chg = mf.mulliken_pop()
        let result = mf
            .call_method0("mulliken_pop")
            .map_err(|e| OracleError::Upstream(format!("mulliken_pop: {}", e)))?;
        let chg_upstream_obj = result
            .get_item(1)
            .map_err(|e| OracleError::Upstream(format!("tuple[1]: {}", e)))?;
        let chg_upstream = np_to_vec_f64(&chg_upstream_obj)?;

        let rust_mol = build_rust_mol(fixture)?;
        let mut rhf = pyscf_scf::RHF::new(rust_mol);
        rhf.kernel()
            .map_err(|e| OracleError::PyscfRs(format!("kernel: {}", e)))?;
        let pop = pyscf_scf::mulliken_pop(&rhf)
            .map_err(|e| OracleError::PyscfRs(format!("mulliken_pop: {}", e)))?;

        if pop.atom_charges.len() != chg_upstream.len() {
            return Err(OracleError::Diff {
                diff: f64::INFINITY,
                tol,
                key: format!(
                    "mulliken_pop[{}] shape (rs={}, upstream={})",
                    fixture,
                    pop.atom_charges.len(),
                    chg_upstream.len()
                ),
            });
        }
        let max_diff = elementwise_max_diff(&pop.atom_charges, &chg_upstream);
        if max_diff > tol {
            return Err(OracleError::Diff {
                diff: max_diff,
                tol,
                key: format!("mulliken_pop[{}]", fixture),
            });
        }
        Ok(())
    }

    // ── Arm 8: Dipole moment (3-vector) ──────────────────────────────────

    fn check_dip_moment(py: Python<'_>, fixture: &str, tol: f64) -> Result<(), OracleError> {
        let mol = build_pyscf_mol(py, fixture)?;
        let scf = py
            .import("pyscf.scf")
            .map_err(|e| OracleError::Upstream(format!("import scf: {}", e)))?;
        let mf = scf
            .call_method1("RHF", (mol,))
            .map_err(|e| OracleError::Upstream(format!("RHF(): {}", e)))?;
        mf.call_method0("kernel")
            .map_err(|e| OracleError::Upstream(format!("kernel: {}", e)))?;
        let dip_upstream_obj = mf
            .call_method0("dip_moment")
            .map_err(|e| OracleError::Upstream(format!("dip_moment: {}", e)))?;
        let dip_upstream = np_to_vec_f64(&dip_upstream_obj)?;

        let rust_mol = build_rust_mol(fixture)?;
        let mut rhf = pyscf_scf::RHF::new(rust_mol);
        rhf.kernel()
            .map_err(|e| OracleError::PyscfRs(format!("kernel: {}", e)))?;
        let dip = pyscf_scf::dip_moment(&rhf)
            .map_err(|e| OracleError::PyscfRs(format!("dip_moment: {}", e)))?;

        if dip.len() != 3 || dip_upstream.len() != 3 {
            return Err(OracleError::Diff {
                diff: f64::INFINITY,
                tol,
                key: format!(
                    "dip_moment[{}] dimension (rs={}, upstream={})",
                    fixture,
                    dip.len(),
                    dip_upstream.len()
                ),
            });
        }
        let max_diff = elementwise_max_diff(&dip, &dip_upstream);
        if max_diff > tol {
            return Err(OracleError::Diff {
                diff: max_diff,
                tol,
                key: format!("dip_moment[{}]", fixture),
            });
        }
        Ok(())
    }

    // ── Arm 9: DFT-04/09 Becke grid coords + weights byte-for-byte ──────────

    /// Parse a `<base>@levelN` fixture name into `(base, level)`.
    fn parse_grid_fixture(fixture: &str) -> (&str, usize) {
        match fixture.split_once("@level") {
            Some((base, lvl)) => (base, lvl.parse::<usize>().unwrap_or(3)),
            None => (fixture, 3),
        }
    }

    fn check_grid_weights(py: Python<'_>, fixture: &str, tol: f64) -> Result<(), OracleError> {
        let (base, level) = parse_grid_fixture(fixture);

        // ── Upstream: dft.gen_grid.Grids with class defaults, sort_grids=False ──
        let mol = build_pyscf_mol(py, base)?;
        let gen_grid = py
            .import("pyscf.dft.gen_grid")
            .map_err(|e| OracleError::Upstream(format!("import gen_grid: {}", e)))?;
        let grids = gen_grid
            .call_method1("Grids", (mol,))
            .map_err(|e| OracleError::Upstream(format!("Grids(): {}", e)))?;
        grids
            .setattr("level", level)
            .map_err(|e| OracleError::Upstream(format!("set level: {}", e)))?;
        // build(sort_grids=False) — match the pyscf-rs unsorted contract.
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs
            .set_item("sort_grids", false)
            .map_err(|e| OracleError::Upstream(format!("kwargs sort_grids: {}", e)))?;
        grids
            .call_method("build", (), Some(&kwargs))
            .map_err(|e| OracleError::Upstream(format!("grids.build: {}", e)))?;
        let coords_upstream = np_to_vec_f64(
            &grids
                .getattr("coords")
                .map_err(|e| OracleError::Upstream(format!("coords attr: {}", e)))?,
        )?;
        let weights_upstream = np_to_vec_f64(
            &grids
                .getattr("weights")
                .map_err(|e| OracleError::Upstream(format!("weights attr: {}", e)))?,
        )?;

        // ── pyscf-rs: pyscf_grids::Grids with class defaults ──
        let rust_mol = build_rust_mol(base)?;
        let mut g = pyscf_grids::Grids::new();
        g.level = level;
        let (coords_rs_rows, weights_rs) = g.build(&rust_mol);
        // Flatten coords row-major to match numpy's (N,3).reshape order.
        let mut coords_rs = Vec::with_capacity(coords_rs_rows.len() * 3);
        for row in &coords_rs_rows {
            coords_rs.extend_from_slice(row);
        }

        // Shape gate.
        if coords_rs.len() != coords_upstream.len() {
            return Err(OracleError::Diff {
                diff: f64::INFINITY,
                tol,
                key: format!(
                    "grid_weights[{}] coords shape (rs={}, upstream={})",
                    fixture,
                    coords_rs.len(),
                    coords_upstream.len()
                ),
            });
        }
        if weights_rs.len() != weights_upstream.len() {
            return Err(OracleError::Diff {
                diff: f64::INFINITY,
                tol,
                key: format!(
                    "grid_weights[{}] weights shape (rs={}, upstream={})",
                    fixture,
                    weights_rs.len(),
                    weights_upstream.len()
                ),
            });
        }

        let coords_diff = elementwise_max_diff(&coords_rs, &coords_upstream);
        if coords_diff > tol {
            return Err(OracleError::Diff {
                diff: coords_diff,
                tol,
                key: format!("grid_weights[{}] coords", fixture),
            });
        }
        let weights_diff = elementwise_max_diff(&weights_rs, &weights_upstream);
        if weights_diff > tol {
            return Err(OracleError::Diff {
                diff: weights_diff,
                tol,
                key: format!("grid_weights[{}] weights", fixture),
            });
        }
        Ok(())
    }

    // ── Arm 10 (DFT-01): RKS total energy ────────────────────────────────
    //
    // Upstream `dft.RKS(mol, xc=...).kernel()` vs pyscf-rs `RKS::new(mol,
    // xc).kernel()`, within `tol` µHartree. The fixture name encodes the XC
    // functional as `<base>@<xc>` (fixtures::xc). f64 ONLY (PYSCF_DTYPE must
    // be unset — the bit-exact path; D-08). SVWN/PBE/B3LYP exercise the
    // LDA / GGA / standard-hybrid (hyb-scaled K) branches respectively.
    fn check_rks_energy(py: Python<'_>, fixture: &str, tol: f64) -> Result<(), OracleError> {
        let xc = fixtures::xc(fixture);
        let mol = build_pyscf_mol(py, fixture)?;
        let dft = py
            .import("pyscf.dft")
            .map_err(|e| OracleError::Upstream(format!("import dft: {}", e)))?;
        let mf = dft
            .call_method1("RKS", (mol,))
            .map_err(|e| OracleError::Upstream(format!("RKS(): {}", e)))?;
        mf.setattr("xc", xc)
            .map_err(|e| OracleError::Upstream(format!("set xc: {}", e)))?;
        let e_upstream: f64 = mf
            .call_method0("kernel")
            .map_err(|e| OracleError::Upstream(format!("kernel: {}", e)))?
            .extract()
            .map_err(|e| OracleError::Upstream(format!("extract f64: {}", e)))?;

        let rust_mol = build_rust_mol(fixture)?;
        let mut ks = pyscf_dft::RKS::new(rust_mol, xc);
        ks.kernel()
            .map_err(|e| OracleError::PyscfRs(format!("RKS kernel: {}", e)))?;
        let e_rs = ks.e_tot;

        let diff = (e_rs - e_upstream).abs();
        if diff > tol {
            return Err(OracleError::Diff {
                diff,
                tol,
                key: format!("rks_energy[{}/{}]", fixture, xc),
            });
        }
        Ok(())
    }

    // ── Arm 11 (DFT-01): UKS total energy (open shell) ───────────────────
    fn check_uks_energy(py: Python<'_>, fixture: &str, tol: f64) -> Result<(), OracleError> {
        let xc = fixtures::xc(fixture);
        let mol = build_pyscf_mol(py, fixture)?;
        let dft = py
            .import("pyscf.dft")
            .map_err(|e| OracleError::Upstream(format!("import dft: {}", e)))?;
        let mf = dft
            .call_method1("UKS", (mol,))
            .map_err(|e| OracleError::Upstream(format!("UKS(): {}", e)))?;
        mf.setattr("xc", xc)
            .map_err(|e| OracleError::Upstream(format!("set xc: {}", e)))?;
        let e_upstream: f64 = mf
            .call_method0("kernel")
            .map_err(|e| OracleError::Upstream(format!("kernel: {}", e)))?
            .extract()
            .map_err(|e| OracleError::Upstream(format!("extract f64: {}", e)))?;

        let rust_mol = build_rust_mol(fixture)?;
        let mut ks = pyscf_dft::UKS::new(rust_mol, xc);
        ks.kernel()
            .map_err(|e| OracleError::PyscfRs(format!("UKS kernel: {}", e)))?;
        let e_rs = ks.e_tot;

        let diff = (e_rs - e_upstream).abs();
        if diff > tol {
            return Err(OracleError::Diff {
                diff,
                tol,
                key: format!("uks_energy[{}/{}]", fixture, xc),
            });
        }
        Ok(())
    }

    // ── Arm 12 (DFT-07): DF-DFT total energy ─────────────────────────────
    // `dft.RKS(mol, xc).density_fit().kernel()` — the Coulomb-J build routes
    // through the Phase 3 pyscf-df crate (D-10 reuse); the Vxc/K stay standard.
    fn check_df_dft_energy(py: Python<'_>, fixture: &str, tol: f64) -> Result<(), OracleError> {
        let xc = fixtures::xc(fixture);
        let mol = build_pyscf_mol(py, fixture)?;
        let dft = py
            .import("pyscf.dft")
            .map_err(|e| OracleError::Upstream(format!("import dft: {}", e)))?;
        let mf = dft
            .call_method1("RKS", (mol,))
            .map_err(|e| OracleError::Upstream(format!("RKS(): {}", e)))?;
        mf.setattr("xc", xc)
            .map_err(|e| OracleError::Upstream(format!("set xc: {}", e)))?;
        // .density_fit() returns the DF-wrapped method object.
        let mf_df = mf
            .call_method0("density_fit")
            .map_err(|e| OracleError::Upstream(format!("density_fit(): {}", e)))?;
        let e_upstream: f64 = mf_df
            .call_method0("kernel")
            .map_err(|e| OracleError::Upstream(format!("kernel: {}", e)))?
            .extract()
            .map_err(|e| OracleError::Upstream(format!("extract f64: {}", e)))?;

        let rust_mol = build_rust_mol(fixture)?;
        let mut ks = pyscf_dft::RKS::new(rust_mol, xc)
            .density_fit(None)
            .map_err(|e| OracleError::PyscfRs(format!("RKS::density_fit: {}", e)))?;
        ks.kernel_df()
            .map_err(|e| OracleError::PyscfRs(format!("RKS kernel_df: {}", e)))?;
        let e_rs = ks.e_tot;

        let diff = (e_rs - e_upstream).abs();
        if diff > tol {
            return Err(OracleError::Diff {
                diff,
                tol,
                key: format!("df_dft_energy[{}/{}]", fixture, xc),
            });
        }
        Ok(())
    }

    // ── Arm 13 (DFT-07 chkfile / ORACLE-08): KS chkfile round-trip ───────
    // KsResult written by pyscf-rs is h5py-readable on the upstream /scf
    // schema (+ the xc/grids metadata); a PySCF DFT chkfile is pyscf-rs
    // -readable. Mirrors `check_chkfile_roundtrip` but for the KS result type.
    fn check_ks_chkfile_roundtrip(
        py: Python<'_>,
        fixture: &str,
        tol: f64,
    ) -> Result<(), OracleError> {
        let xc = fixtures::xc(fixture);

        // Direction (a): PySCF DFT writes → pyscf-rs reads the /scf block.
        let tmp_a = tempfile::NamedTempFile::new()?;
        let path_a = tmp_a.path();
        {
            let mol = build_pyscf_mol(py, fixture)?;
            let dft = py
                .import("pyscf.dft")
                .map_err(|e| OracleError::Upstream(format!("import dft: {}", e)))?;
            let mf = dft
                .call_method1("RKS", (mol,))
                .map_err(|e| OracleError::Upstream(format!("RKS(): {}", e)))?;
            mf.setattr("xc", xc)
                .map_err(|e| OracleError::Upstream(format!("set xc: {}", e)))?;
            mf.setattr(
                "chkfile",
                path_a
                    .to_str()
                    .ok_or_else(|| OracleError::Upstream("path_a not utf8".into()))?,
            )
            .map_err(|e| OracleError::Upstream(format!("setattr chkfile: {}", e)))?;
            mf.call_method0("kernel")
                .map_err(|e| OracleError::Upstream(format!("kernel: {}", e)))?;
        }
        // pyscf-rs reads the /scf block PySCF DFT just wrote (the SCF schema is
        // shared; the KS xc/grids metadata is pyscf-rs's extension and is not
        // written by upstream, so we read the SCF block here for the e_tot
        // cross-check).
        let scf_loaded = pyscf_scf::load_scf_from_file(path_a)
            .map_err(|e| OracleError::PyscfRs(format!("load_scf_from_file: {}", e)))?;
        let pyscf_lib = py
            .import("pyscf.lib.chkfile")
            .map_err(|e| OracleError::Upstream(format!("import pyscf.lib.chkfile: {}", e)))?;
        let chk_data = pyscf_lib
            .call_method1(
                "load",
                (
                    path_a
                        .to_str()
                        .ok_or_else(|| OracleError::Upstream("path_a not utf8".into()))?,
                    "scf",
                ),
            )
            .map_err(|e| OracleError::Upstream(format!("chkfile.load: {}", e)))?;
        let e_upstream: f64 = chk_data
            .get_item("e_tot")
            .map_err(|e| OracleError::Upstream(format!("get e_tot: {}", e)))?
            .extract()
            .map_err(|e| OracleError::Upstream(format!("extract e_tot: {}", e)))?;
        let diff_a = (scf_loaded.e_tot.0 - e_upstream).abs();
        if diff_a > tol {
            return Err(OracleError::Diff {
                diff: diff_a,
                tol,
                key: format!("ks_chkfile_roundtrip[A,{}]", fixture),
            });
        }

        // Direction (b): pyscf-rs writes a KsResult → h5py reads /scf/e_tot AND
        // the KS xc/grids metadata (the h5py-readability seal on the extended
        // schema). We build a representative KsResult from the loaded SCF block.
        let tmp_b = tempfile::NamedTempFile::new()?;
        let path_b = tmp_b.path();
        let ks_result = pyscf_dft::KsResult::new(scf_loaded, xc, pyscf_dft::GridsMeta::default());
        pyscf_dft::dump_ks_to_file(path_b, "{}", &ks_result)
            .map_err(|e| OracleError::PyscfRs(format!("dump_ks_to_file: {}", e)))?;

        // h5py opens the pyscf-rs-written chkfile and reads /scf/e_tot + /scf/xc.
        let h5py = py
            .import("h5py")
            .map_err(|e| OracleError::Upstream(format!("import h5py: {}", e)))?;
        let f = h5py
            .call_method1(
                "File",
                (
                    path_b
                        .to_str()
                        .ok_or_else(|| OracleError::Upstream("path_b not utf8".into()))?,
                    "r",
                ),
            )
            .map_err(|e| OracleError::Upstream(format!("h5py.File: {}", e)))?;
        let scf_grp = f
            .get_item("scf")
            .map_err(|e| OracleError::Upstream(format!("h5py['scf']: {}", e)))?;
        // /scf/e_tot reads back as a scalar (`dataset[()]` — h5py-readable on
        // the upstream schema). The empty tuple is the scalar index.
        let empty_idx = pyo3::types::PyTuple::empty(py);
        let e_dataset = scf_grp
            .get_item("e_tot")
            .map_err(|e| OracleError::Upstream(format!("h5py scf/e_tot: {}", e)))?;
        let e_h5: f64 = e_dataset
            .call_method1("__getitem__", (empty_idx,))
            .map_err(|e| OracleError::Upstream(format!("read e_tot scalar: {}", e)))?
            .extract()
            .map_err(|e| OracleError::Upstream(format!("extract e_tot h5: {}", e)))?;
        // /scf/xc reads back as the DFT metadata (the schema extension).
        let _xc_h5 = scf_grp
            .get_item("xc")
            .map_err(|e| OracleError::Upstream(format!("h5py scf/xc missing: {}", e)))?;
        f.call_method0("close")
            .map_err(|e| OracleError::Upstream(format!("h5py close: {}", e)))?;

        let diff_b = (ks_result.scf.e_tot.0 - e_h5).abs();
        if diff_b > tol {
            return Err(OracleError::Diff {
                diff: diff_b,
                tol,
                key: format!("ks_chkfile_roundtrip[B,{}]", fixture),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_method_returns_unknown_method_error_without_python_feature() {
        // Pre-dispatch guard: unknown method short-circuits before any
        // Python work, so this branch is exercised in every build mode.
        let r = run_oracle_check("nonexistent_method", "h2o_ccpvdz", 1e-6);
        match r {
            Err(OracleError::UnknownMethod(name)) => {
                assert_eq!(name, "nonexistent_method");
            }
            other => panic!(
                "expected UnknownMethod for nonexistent_method, got {:?}",
                other
            ),
        }
    }

    #[cfg(not(feature = "python"))]
    #[test]
    fn known_method_without_python_feature_returns_python_feature_not_enabled() {
        let r = run_oracle_check("scf_rhf_energy", "h2o_ccpvdz", 1e-6);
        assert!(
            matches!(r, Err(OracleError::PythonFeatureNotEnabled)),
            "expected PythonFeatureNotEnabled, got {:?}",
            r
        );
    }

    #[test]
    fn known_methods_list_has_all_arms() {
        // 8 SCF/DF arms (Phase 3) + grid_weights (04-04) + rks_energy +
        // uks_energy (04-06) + df_dft_energy + ks_chkfile_roundtrip (04-08) = 13,
        // + 5 Phase-5 MP2 arms (05-01: mp2_rmp2_energy, mp2_ump2_energy,
        // dfmp2_energy, dfmp2_native_energy, mp2_rdm) = 18,
        // + 6 Phase-6 CCSD arms (06-11: ccsd_rccsd_energy, ccsd_uccsd_energy,
        // ccsd_dfccsd_energy, ccsd_lambda, ccsd_rdm1, ccsd_rdm2) = 24,
        // + 8 Phase-7 grad/geomopt arms (07-10: nuc_grad_rhf, nuc_grad_uhf,
        // nuc_grad_rks, nuc_grad_uks, nuc_grad_mp2, nuc_grad_ccsd, nuc_grad_ecp,
        // geomopt_h2o) = 32.
        assert_eq!(KNOWN_METHODS.len(), 32);
        assert!(KNOWN_METHODS.contains(&"scf_rhf_energy"));
        assert!(KNOWN_METHODS.contains(&"scf_uhf_energy"));
        assert!(KNOWN_METHODS.contains(&"scf_diis_iter_count"));
        assert!(KNOWN_METHODS.contains(&"scf_init_guess"));
        assert!(KNOWN_METHODS.contains(&"df_hf_energy"));
        assert!(KNOWN_METHODS.contains(&"chkfile_roundtrip"));
        assert!(KNOWN_METHODS.contains(&"mulliken_pop"));
        assert!(KNOWN_METHODS.contains(&"dip_moment"));
        assert!(KNOWN_METHODS.contains(&"grid_weights"));
        assert!(KNOWN_METHODS.contains(&"rks_energy"));
        assert!(KNOWN_METHODS.contains(&"uks_energy"));
        assert!(KNOWN_METHODS.contains(&"df_dft_energy"));
        assert!(KNOWN_METHODS.contains(&"ks_chkfile_roundtrip"));
        // Phase 5 MP2 arms (05-01).
        assert!(KNOWN_METHODS.contains(&"mp2_rmp2_energy"));
        assert!(KNOWN_METHODS.contains(&"mp2_ump2_energy"));
        assert!(KNOWN_METHODS.contains(&"dfmp2_energy"));
        assert!(KNOWN_METHODS.contains(&"dfmp2_native_energy"));
        assert!(KNOWN_METHODS.contains(&"mp2_rdm"));
        // Phase 6 CCSD arms (06-11).
        assert!(KNOWN_METHODS.contains(&"ccsd_rccsd_energy"));
        assert!(KNOWN_METHODS.contains(&"ccsd_uccsd_energy"));
        assert!(KNOWN_METHODS.contains(&"ccsd_dfccsd_energy"));
        assert!(KNOWN_METHODS.contains(&"ccsd_lambda"));
        assert!(KNOWN_METHODS.contains(&"ccsd_rdm1"));
        assert!(KNOWN_METHODS.contains(&"ccsd_rdm2"));
        // Phase 7 grad/geomopt arms (07-10).
        assert!(KNOWN_METHODS.contains(&"nuc_grad_rhf"));
        assert!(KNOWN_METHODS.contains(&"nuc_grad_uhf"));
        assert!(KNOWN_METHODS.contains(&"nuc_grad_rks"));
        assert!(KNOWN_METHODS.contains(&"nuc_grad_uks"));
        assert!(KNOWN_METHODS.contains(&"nuc_grad_mp2"));
        assert!(KNOWN_METHODS.contains(&"nuc_grad_ccsd"));
        assert!(KNOWN_METHODS.contains(&"nuc_grad_ecp"));
        assert!(KNOWN_METHODS.contains(&"geomopt_h2o"));
    }
}

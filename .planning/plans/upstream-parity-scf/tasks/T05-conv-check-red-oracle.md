# T05 — Final diagonalisation: red oracle test

**Goal.** A test against upstream PySCF that fails today because the port
reports level-shifted orbital energies.

**Upstream behaviour** (`pyscf/scf/hf.py:211-232`): after the SCF loop
converges, upstream diagonalises the plain Fock matrix once more (no level
shift, no DIIS), recomputes occupations, density and energy, and re-tests
convergence. The port stops at the converged cycle
(`crates/pyscf-pbc-scf/src/kscf.rs:207-220`).

## Read first

- `crates/pyscf-pbc-dft/tests/scf_ovlp_smearing_oracle.rs` — copy its
  structure (imports, `mod common;`, `ORACLE_PY`, `run_python`, `cell_args`).
- `crates/pyscf-pbc-dft/tests/common/mod.rs` — `diamond()`, `GATE`,
  `oracle_python`, `run_python`, `cell_args`.

## Do

Create `crates/pyscf-pbc-dft/tests/kscf_conv_check_oracle.rs`:

```rust
//! The final diagonalisation after convergence (`scf/hf.py:211-232`,
//! `conv_check`): KRKS/PBE on diamond with a level shift, against live
//! upstream PySCF 2.12.1. Without that final step the reported orbital
//! energies are the level-shifted ones.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release \
//!     --test kscf_conv_check_oracle -- --ignored --nocapture
//! ```

mod common;

use common::{GATE, cell_args, diamond, oracle_python, run_python};
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_gto::make_kpts_default;
use pyscf_pbc_scf::KScfConfig;

const NK: [usize; 3] = [2, 2, 2];
const SHIFT: f64 = 0.2;

const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto, dft
from pyscf.dft import libxc

a_json, xyz_json, sym_json, spin, charge, nk_json, shift = sys.argv[1:8]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = 'gth-szv'
c.pseudo = 'gth-pade'
c.unit = 'Bohr'
c.spin = int(spin)
c.charge = int(charge)
c.verbose = 0
c.build()
mf = dft.KRKS(c, c.make_kpts(json.loads(nk_json)), xc='pbe')
mf._numint.libxc = libxc
mf.level_shift = float(shift)
mf.conv_tol = 1e-9
mf.max_cycle = 100
e = mf.kernel()
print(json.dumps({
    'version': __import__('pyscf').__version__,
    'converged': bool(mf.converged),
    'mesh': [int(x) for x in c.mesh],
    'e_tot': float(e),
    'mo_energy_gamma': np.asarray(mf.mo_energy[0]).tolist(),
}))
"#;

#[test]
#[ignore = "oracle: set PYSCF_ORACLE_VENV (upstream PySCF 2.12.1)"]
fn level_shifted_krks_reports_unshifted_orbital_energies() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} unset — skipping");
        return;
    };
    let cell = diamond();
    let args = cell_args(
        &cell,
        &[serde_json::to_string(&NK.to_vec()).expect("json"), SHIFT.to_string()],
    );
    let want = run_python(&py, ORACLE_PY, &args);
    assert_eq!(want["version"].as_str(), Some("2.12.1"), "oracle must be vendored 2.12.1");
    assert_eq!(want["converged"].as_bool(), Some(true), "upstream did not converge");
    let mesh: Vec<usize> = want["mesh"]
        .as_array()
        .expect("mesh")
        .iter()
        .map(|v| v.as_u64().expect("mesh int") as usize)
        .collect();
    assert_eq!(mesh, cell.try_mesh().expect("mesh").to_vec(), "cell.mesh differs");

    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let mut mf = Krks::new(cell, &kpts, "pbe").expect("KRKS");
    let got = mf
        .kernel(&KScfConfig {
            conv_tol: 1e-9,
            max_cycle: 100,
            level_shift: SHIFT,
            ..KScfConfig::default()
        })
        .expect("KRKS");
    assert!(got.converged, "port did not converge in {} cycles", got.cycles);

    let e_tot = want["e_tot"].as_f64().expect("e_tot");
    let de = (got.e_tot - e_tot).abs();
    let eps: Vec<f64> = want["mo_energy_gamma"]
        .as_array()
        .expect("mo_energy")
        .iter()
        .map(|v| v.as_f64().expect("f64"))
        .collect();
    let d_eps = got.mo_energy[0]
        .iter()
        .zip(&eps)
        .fold(0.0_f64, |a, (g, w)| a.max((g - w).abs()));
    println!(
        "diamond 2x2x2 PBE shift {SHIFT}: e_tot |Δ| {de:.3e}; Γ mo_energy max|Δ| {d_eps:.3e}; cycles {}",
        got.cycles
    );
    assert!(de < 1e-7, "e_tot |Δ| {de:e}");
    assert!(d_eps < 1e-5, "Γ mo_energy max|Δ| {d_eps:e} — level shift still in the reported orbital energies?");
}
```

If `Krks::kernel` takes `&self` instead of `&mut self`, drop the `mut`.

## Verify

ORACLE form: `-p pyscf-pbc-dft --test kscf_conv_check_oracle`.
Expected NOW: the test **FAILS** at the last assertion with
`Γ mo_energy max|Δ|` about `2e-1`. The `e_tot` assertion must pass.
A failure at the last assertion is the correct result of this task.

## If it fails differently

- Printed `SKIP`: the variable `PYSCF_ORACLE_VENV=1` was not set.
- `e_tot` assertion fails: STOP and report (that is a different problem).
- Python error about `h5py` or a wrong version: the test must be run from
  the repository root through cargo; do not run the script by hand.

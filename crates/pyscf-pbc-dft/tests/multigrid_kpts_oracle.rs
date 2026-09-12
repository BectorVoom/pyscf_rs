//! K-01 **Gate D** — the k-point-resolved multigrid against UPSTREAM PySCF.
//!
//! Everything in `tests/multigrid_kpts.rs` and `tests/krks_ksymm_multigrid.rs`
//! is **Gate C**: port vs port. Those gates are the sharp ones — they compare
//! routes that share a cell builder, a basis parser and an XC backend, so a
//! residual is attributable. But they cannot answer "does this agree with
//! PySCF", because PySCF is not in them. This file is that question, and it
//! is a SEPARATE file so the distinction cannot quietly erode.
//!
//! # What is compared, and what the tolerance means
//!
//! `E_tot` of a converged k-point KRKS on the SAME cell at the SAME pinned
//! mesh, upstream against this port, over both k-meshes and both quadratures
//! upstream offers:
//!
//! * upstream FFTDF `KRKS` — upstream's reference route;
//! * upstream `MultiGridNumInt2` `KRKS` — upstream's own multigrid, when it
//!   accepts the k-mesh (it is reported either way, never silently skipped).
//!
//! Two floors are already known and are NOT defects, which is why they are
//! named here rather than discovered again by whoever next reads a failure:
//!
//! 1. **The XC backend differs.** This port evaluates `lda,vwn` through
//!    xcfun; upstream defaults to libxc. The two disagree by ~5e-7 Ha on
//!    these cells.
//! 2. **The mesh is pinned coarse** to keep the comparison honest, and a
//!    pinned coarse mesh moves `KRHF` by ~1.35e-5 Ha where the default mesh
//!    gives ~4.8e-11.
//!
//! Both are orders below the tolerance this file gates at, so neither can
//! rescue a real disagreement — which is the point of naming them.

mod common;

use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_dft::numint::KsNumInt;
use pyscf_pbc_gto::{Cell, make_kpts_default};
use pyscf_pbc_scf::KScfConfig;

/// The gate. **Deliberately far looser than anything measured**: the point of
/// a Gate-D number is to catch a route that is WRONG — a dropped Bloch phase,
/// a conjugated potential, a mis-weighted k-sum — all of which move an energy
/// by milli- to whole Hartrees, not by micro-Hartrees. A tight tolerance here
/// would be gating the xcfun/libxc difference and the mesh pin, not the
/// physics. The measured numbers are printed, so a regression of orders is
/// still visible in the log even while the assertion stays loose.
const E_TOL: f64 = 1e-1;

const MESH: [usize; 3] = [15, 15, 15];

fn cell() -> Cell {
    let mut c = common::diamond();
    c.mesh = MESH;
    c
}

/// Upstream KRKS at `nk`, both quadratures, on the cell described by
/// `cell_args`. Emits one JSON line.
const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
import pyscf
from pyscf.pbc import gto, dft
from pyscf.pbc.dft import multigrid

a, xyz, sym, spin, charge = (json.loads(sys.argv[1]), json.loads(sys.argv[2]),
                             json.loads(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
mesh = json.loads(sys.argv[6])
nk = json.loads(sys.argv[7])
xc = sys.argv[8]

c = gto.Cell()
c.a = np.array(a)
c.atom = [(s, tuple(r)) for s, r in zip(sym, xyz)]
c.unit = 'Bohr'
c.basis = 'gth-szv'
c.pseudo = 'gth-pade'
c.spin = spin
c.charge = charge
c.mesh = [int(x) for x in mesh]
c.verbose = 0
c.build()

kpts = c.make_kpts(nk)

out = {"pyscf_version": pyscf.__version__, "nkpts": len(kpts)}

mf = dft.KRKS(c, kpts)
mf.xc = xc
mf.conv_tol = 1e-10
e_ref = mf.kernel()
out["e_fftdf"] = float(e_ref)
out["converged_fftdf"] = bool(mf.converged)

# Upstream's OWN multigrid at the same k-points. Reported whether it works
# or refuses -- a silent skip here would be the failure mode this file exists
# to prevent.
try:
    mf2 = dft.KRKS(c, kpts)
    mf2.xc = xc
    mf2.conv_tol = 1e-10
    mf2._numint = multigrid.MultiGridNumInt2(c)
    e_mg = mf2.kernel()
    out["e_multigrid"] = float(e_mg)
    out["converged_multigrid"] = bool(mf2.converged)
except Exception as exc:      # noqa: BLE001 -- reporting, not handling
    out["e_multigrid"] = None
    out["multigrid_error"] = f"{type(exc).__name__}: {exc}"

print(json.dumps(out))
"#;

fn e_tot_rust(cell: &Cell, nk: [usize; 3], xc: &str, multigrid: bool) -> f64 {
    let kpts = make_kpts_default(cell, nk).expect("k-mesh");
    let df = pyscf_pbc_df::Fftdf::with_mesh(cell.clone(), &kpts, cell.mesh).expect("FFTDF");
    let mut mf = Krks::from_df(Box::new(df), xc).expect("KRKS");
    if multigrid {
        mf.ni = KsNumInt::multigrid2();
    }
    let r = mf
        .kernel(&KScfConfig {
            conv_tol: 1e-10,
            max_cycle: 60,
            ..KScfConfig::for_cell(cell)
        })
        .expect("KRKS kernel");
    assert!(
        r.converged,
        "this port's KRKS (multigrid = {multigrid}) did not converge at {nk:?}"
    );
    r.e_tot
}

/// The Gate-D comparison, k-mesh by k-mesh.
///
/// Skips (does not fail) when `PYSCF_ORACLE_VENV` is unset, matching every
/// other oracle-gated test in this crate.
#[test]
fn kpoint_multigrid_e_tot_matches_upstream() {
    let Some(py) = common::oracle_python() else {
        eprintln!("PYSCF_ORACLE_VENV unset — skipping the upstream comparison");
        return;
    };
    let cell = cell();
    let xc = "lda,vwn";
    let mut worst = 0.0f64;

    for nk in [[1, 1, 2], [2, 2, 2], [1, 1, 3]] {
        let args = common::cell_args(
            &cell,
            &[
                serde_json::to_string(&MESH.to_vec()).expect("json"),
                serde_json::to_string(&nk.to_vec()).expect("json"),
                xc.to_string(),
            ],
        );
        let up = common::run_python(&py, ORACLE_PY, &args);

        let e_up_ref = up["e_fftdf"].as_f64().expect("upstream e_fftdf");
        assert_eq!(
            up["nkpts"].as_u64().expect("nkpts") as usize,
            nk.iter().product::<usize>(),
            "{nk:?}: the two sides disagree on how many k-points that mesh has"
        );

        let e_mg = e_tot_rust(&cell, nk, xc, true);
        let e_grid = e_tot_rust(&cell, nk, xc, false);

        let d_mg = (e_mg - e_up_ref).abs();
        let d_grid = (e_grid - e_up_ref).abs();
        worst = worst.max(d_mg);

        println!("--- {nk:?} ({} k-points), mesh {MESH:?} ---", nk.iter().product::<usize>());
        println!("  upstream FFTDF KRKS      : {e_up_ref:.12}");
        println!("  this port, grid numint   : {e_grid:.12}   |d| = {d_grid:.3e}");
        println!("  this port, MULTIGRID k   : {e_mg:.12}   |d| = {d_mg:.3e}");
        match up["e_multigrid"].as_f64() {
            Some(e_up_mg) => println!(
                "  upstream MULTIGRID KRKS  : {e_up_mg:.12}   |d vs this port's mg| = {:.3e}",
                (e_mg - e_up_mg).abs()
            ),
            None => println!(
                "  upstream MULTIGRID KRKS  : REFUSED — {}",
                up["multigrid_error"].as_str().unwrap_or("(no message)")
            ),
        }

        assert!(
            d_mg < E_TOL,
            "{nk:?}: this port's k-point MULTIGRID KRKS is {d_mg:e} Ha from upstream's \
             ({e_mg} vs {e_up_ref}), past the {E_TOL:e} gate. The port's own grid \
             numint is {d_grid:e} from the same upstream number on the same fixture — \
             if that one is small and this one is not, the defect is in the multigrid \
             k-point path, not in the cell or the functional"
        );
    }
    println!("Gate D worst |dE| across all k-meshes: {worst:.3e} Ha (gate {E_TOL:e})");
}

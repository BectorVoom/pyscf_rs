//! `FFTDF.get_pp` with upstream's reciprocal-space non-local part
//! (`pbc/df/fft.py:114-176`) on KTaO3 — projector channels `l = 0, 1, 2`,
//! up to three projectors, complex k-points — against live upstream
//! PySCF 2.12.1 at a coarse mesh.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-df --release \
//!     --test pp_gspace_oracle -- --ignored --nocapture
//! ```

mod common;

use common::{GATE, cell_args, max_dev, oracle_python, run_python};
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_df::fftdf::Fftdf;
use pyscf_pbc_df::traits::PeriodicDf;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};

pub const BASIS: &str = "gth-szv-molopt-sr";
pub const NK: [usize; 3] = [3, 3, 1];
pub const MESH: [usize; 3] = [21, 21, 21];

/// Cubic KTaO3, Bohr.
pub fn ktao3(basis: &str) -> Cell {
    let a = 7.54_f64;
    let h = 0.5 * a;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("K".into(), [0.0, 0.0, 0.0]),
                ("Ta".into(), [h, h, h]),
                ("O".into(), [h, h, 0.0]),
                ("O".into(), [h, 0.0, h]),
                ("O".into(), [0.0, h, h]),
            ]),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]),
        pseudo: Some("gth-pbe".into()),
        ..Default::default()
    })
    .expect("KTaO3 cell must build")
}

pub const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
from pyscf import gto as mgto
from pyscf.pbc import gto, df
from pyscf.pbc.df import ft_ao
from pyscf.pbc.gto import pseudo

a_json, xyz_json, sym_json, basis, nk_json, mesh_json, what, kidx = sys.argv[1:9]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
c.pseudo = 'gth-pbe'
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mesh = json.loads(mesh_json)
kpt = kpts[int(kidx)]

if what == 'pp':
    mydf = df.FFTDF(c, kpts)
    mydf.mesh = mesh
    mats = np.asarray(mydf.get_pp(kpts))
elif what == 'aokG':
    mats = ft_ao.ft_ao(c, c.get_Gv(mesh), kpt=kpt)[None]
elif what == 'proj':
    Gv = c.get_Gv(mesh)
    Gk = Gv + kpt
    G_rad = np.linalg.norm(Gk, axis=1)
    SI = c.get_SI(Gv)
    fakemol = mgto.Mole()
    fakemol._atm = np.zeros((1, mgto.ATM_SLOTS), dtype=np.int32)
    fakemol._bas = np.zeros((1, mgto.BAS_SLOTS), dtype=np.int32)
    ptr = mgto.PTR_ENV_START
    fakemol._env = np.zeros(ptr + 10)
    fakemol._bas[0, mgto.NPRIM_OF] = 1
    fakemol._bas[0, mgto.NCTR_OF] = 1
    fakemol._bas[0, mgto.PTR_EXP] = ptr + 3
    fakemol._bas[0, mgto.PTR_COEFF] = ptr + 4
    rows = []
    for ia in range(c.natm):
        symb = c.atom_symbol(ia)
        if symb not in c._pseudo:
            continue
        for l, (rl, nl, hl) in enumerate(c._pseudo[symb][5:]):
            if nl > 0:
                fakemol._bas[0, mgto.ANG_OF] = l
                fakemol._env[ptr + 3] = .5 * rl**2
                fakemol._env[ptr + 4] = rl**(l + 1.5) * np.pi**1.25
                part = fakemol.eval_gto('GTOval', Gk)
                for i in range(nl):
                    qkl = pseudo.pp._qli(G_rad * rl, l, i)
                    for m in range(2 * l + 1):
                        rows.append(SI[ia].conj() * part[:, m] * qkl)
    mats = np.asarray(rows)[None]
else:
    raise SystemExit('unknown quantity ' + what)

mats = np.asarray(mats, dtype=complex)
print(json.dumps({'version': __import__('pyscf').__version__,
                  'shape': list(mats.shape),
                  're': np.real(mats).ravel().tolist(),
                  'im': np.imag(mats).ravel().tolist()}))
"#;

pub fn oracle(
    cell: &Cell,
    basis: &str,
    mesh: [usize; 3],
    what: &str,
    kidx: usize,
) -> Option<serde_json::Value> {
    let py = oracle_python()?;
    let args = cell_args(
        cell,
        &[
            basis.to_string(),
            serde_json::to_string(&NK.to_vec()).expect("json"),
            serde_json::to_string(&mesh.to_vec()).expect("json"),
            what.to_string(),
            kidx.to_string(),
        ],
    );
    let v = run_python(&py, ORACLE_PY, &args);
    assert_eq!(
        v["version"].as_str(),
        Some("2.12.1"),
        "oracle must be vendored 2.12.1"
    );
    Some(v)
}

#[test]
#[ignore = "oracle: set PYSCF_ORACLE_VENV (upstream PySCF 2.12.1)"]
fn get_pp_matches_upstream_on_ktao3_at_a_coarse_mesh() {
    let cell = ktao3(BASIS);
    let Some(want) = oracle(&cell, BASIS, MESH, "pp", 0) else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let df = Fftdf::with_mesh(cell, &kpts, MESH).expect("FFTDF");
    let got = df.get_pp(&kpts).expect("get_pp");
    let w = max_dev(&got, &want);
    println!("KTaO3 mesh 21 get_pp max|delta| vs upstream = {w:e}");
    assert!(w < 1e-10, "get_pp deviates from upstream by {w:e}");
}

const MESH_SMALL: [usize; 3] = [5, 5, 5];

/// Compare a planar `(re, im)` pair with the oracle payload.
fn planar_dev(re: &[f64], im: &[f64], want: &serde_json::Value) -> f64 {
    let t = pyscf_algebra::CTensor::from_planes(re.to_vec(), im.to_vec());
    max_dev(&[t], want)
}

#[test]
#[ignore = "oracle: set PYSCF_ORACLE_VENV (upstream PySCF 2.12.1)"]
fn projector_table_matches_upstream() {
    let cell = ktao3(BASIS);
    let kidx = 4; // a complex k-point of the 3x3x1 mesh
    let Some(want) = oracle(&cell, BASIS, MESH_SMALL, "proj", kidx) else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH_SMALL)).expect("Gv");
    let t = pyscf_pbc_df::pp_gspace::proj_tables(&cell).expect("tables");
    // K: 2 + 2*3 = 8, Ta: 3 + 2*3 + 2*5 = 19, 3 O: 1 each
    assert_eq!(t.rows.len(), 8 + 19 + 3, "projector row count");
    let (re, im) =
        pyscf_pbc_df::pp_gspace::proj_values_host(&cell, &t, &gv, kpts[kidx]).expect("values");
    let w = planar_dev(&re, &im, &want);
    println!("projector table max|delta| vs upstream = {w:e}");
    assert!(w < 1e-12, "projector table deviates by {w:e}");
}

#[test]
#[ignore = "oracle: set PYSCF_ORACLE_VENV (upstream PySCF 2.12.1)"]
fn host_ft_ao_matches_upstream_including_f_shells() {
    for basis in [BASIS, "gth-dzvp-molopt-sr"] {
        let cell = ktao3(basis);
        let kidx = 4;
        let Some(want) = oracle(&cell, basis, MESH_SMALL, "aokG", kidx) else {
            eprintln!("SKIP: {GATE} is not set");
            return;
        };
        let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
        let gv = pyscf_pbc_gto::get_gv(&cell, Some(MESH_SMALL)).expect("Gv");
        let (re, im) = pyscf_pbc_df::ft_ao::ft_ao_kpt(&cell.mol, &gv, kpts[kidx]).expect("ft_ao");
        let w = planar_dev(&re, &im, &want);
        println!("{basis}: ft_ao max|delta| vs upstream = {w:e}");
        assert!(w < 1e-11, "{basis}: ft_ao deviates by {w:e}");
    }
}

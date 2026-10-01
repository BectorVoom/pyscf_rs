//! **The Y/Ta band gate.** `Krhf::get_bands` against live upstream PySCF
//! 2.12.1 on a cell carrying the two transition metals of YTa7O19.
//!
//! # Why a substitute cell
//!
//! YTa7O19 is 27 atoms; a converged KRHF plus `get_bands` on it is not a
//! session-budget calculation. The question — "are this port's band energies
//! BIT-EXACT against upstream?" — does not depend on cell size. It depends on
//! whether the band chain (`get_hcore` + the plane-wave lattice sum + the
//! `kpts_band` J/K route + a per-k eigendecomposition) reproduces upstream's
//! summation order bit for bit. A two-atom cell runs every one of those stages.
//!
//! What this adds over `krhf_bands_oracle.rs` (He / `sto-3g`, one AO,
//! all-electron) is the part specific to YTa7O19: the ELEMENTS. Y and Ta bring
//! a GTH pseudopotential with `q11` / `q13` valence charges, `l = 2` projector
//! channels, and d shells — none of which the He gate touches. Structure is
//! CsCl-type (simple cubic, Y at the origin, Ta at the body centre,
//! `a0 = 7 Bohr`), the smallest cell holding both elements. 20 AOs, 24 valence
//! electrons. `gth-szv.dat` has no Y or Ta, so the basis is
//! `gth-szv-molopt-sr` — see `pyscf-gto/tests/gth_molopt_transition_metals.rs`.
//!
//! # Why the density comes from UPSTREAM, not from a Rust SCF
//!
//! The first version of this file converged both sides independently and they
//! landed `8.4e-2` Ha apart. That was NOT an integral defect: measured on
//! 2026-09-20, this port's `hcore` matches upstream to `8.9e-13` and its
//! electronic energy ON UPSTREAM'S OWN DENSITY to `4.3e-14`. The cause is the
//! fixture — upstream's converged k=0 HOMO/LUMO are
//! `0.5341607899080156` / `0.53416079020266`, a gap of **2.9e-10 Ha**. CsCl
//! YTa is an intermetallic whose frontier d levels are numerically degenerate,
//! so the aufbau fill at that k-point is arbitrary and the two codes each
//! converged tightly to a DIFFERENT stationary point.
//!
//! Comparing bands across two such solutions measures the occupation choice,
//! not the band code. So the density is taken from upstream and handed to
//! `Krhf::get_bands` directly. That is the discipline
//! `krhf-coarse-mesh-diverges` records — drive the port from upstream's own
//! mean field — and it also removes the Rust SCF, which was ~60 of the failed
//! run's 64 minutes.
//!
//! # Staged assertions, cheapest and most localising first
//!
//! 1. `mesh` — neither side pins it; both must DERIVE the same grid.
//! 2. `nao`, `nelectron` — the basis block and the GTH q-channel.
//! 3. `e_nuc` — the geometry and the pseudopotential charge.
//! 4. `hcore` — the one-electron + GTH projector path, density-free.
//! 5. `energy_elec` on upstream's density — the J/K path.
//! 6. The band eigenvalues themselves.
//!
//! A failure therefore names its own stage instead of blaming the bands.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-scf --release \
//!     --test krhf_bands_oracle_yta -- --ignored --nocapture
//! ```

mod common;

use common::{GATE, cell_args, oracle_python, run_python};
use pyscf_algebra::CTensor;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_scf::{KOverrideHooks, Krhf};

const A0: f64 = 7.0;
const NK: [usize; 3] = [1, 1, 2];

/// Gate threshold. `band-energies-are-never-bitwise-identical`: the band chain
/// runs Rust/CubeCL summation order against upstream's C/BLAS, so ~1e-11 is
/// the measured floor even on a one-AO all-electron cell. The BITWISE counts
/// are printed, not asserted — they are the answer to the question, and
/// asserting them would be designing the test to fail.
const BAND_TOL: f64 = 1e-9;

fn yta_cscl() -> Cell {
    let h = A0 / 2.0;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("Y".into(), [0.0, 0.0, 0.0]),
                ("Ta".into(), [h, h, h]),
            ]),
            basis: BasisInput::Name("gth-szv-molopt-sr".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[A0, 0.0, 0.0], [0.0, A0, 0.0], [0.0, 0.0, A0]]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("YTa cell must build")
}

/// `mo_coeff` is emitted TRANSPOSED so that index `i * nao + mu` is `C[mu, i]`,
/// the layout `krdm::make_rdm1` reads.
///
/// `energy_elec` returns `(e1 + e_coul, e_coul)` — the FIRST value is already
/// the total electronic energy (`pyscf/pbc/scf/khf.py:266`). Adding `e_coul`
/// to it double-counts the Coulomb term; the Rust side uses the same
/// convention, so `e_elec + e_nuc` is the total on both sides.
const BAND_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto, scf

a_json, xyz_json, sym_json, basis, pseudo, nk_json, kband_json = sys.argv[1:8]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
c.pseudo = pseudo
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mf = scf.KRHF(c, kpts)
# Upstream's default MINAO guess raises IndexError for GTH-pseudo Ta
# (scf/hf.py:415, minao_basis), so this cell cannot use it.
mf.init_guess = '1e'
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
mf.max_cycle = 100
e = mf.kernel()

dm = mf.make_rdm1()
e_elec, e_coul = mf.energy_elec(dm)
h1e = mf.get_hcore(cell=c, kpts=kpts)

# Band k-points arrive as ABSOLUTE cartesian numbers computed on the Rust side,
# so the scaled->absolute conversion is not transcribed twice.
kband = np.asarray(json.loads(kband_json), dtype=float)
e_band, _ = mf.get_bands(kband)

def split(a):
    a = np.asarray(a)
    return {'re': a.real.ravel().tolist(), 'im': a.imag.ravel().tolist()}

print(json.dumps({
    'version': __import__('pyscf').__version__,
    'converged': bool(mf.converged),
    'e_tot': float(e), 'e_nuc': float(c.energy_nuc()),
    'e_elec': float(e_elec), 'e_coul': float(e_coul),
    'nao': int(c.nao_nr()), 'nelectron': int(c.nelectron),
    'mesh': [int(x) for x in c.mesh],
    'mo_occ': [np.asarray(o).ravel().tolist() for o in mf.mo_occ],
    'mo_coeff_T': [split(np.asarray(m).T) for m in mf.mo_coeff],
    'hcore': [split(h) for h in h1e],
    'e_band': [np.asarray(x).ravel().tolist() for x in e_band],
    'homo_lumo': [[float(np.sort(x)[c.nelectron//2 - 1]),
                   float(np.sort(x)[c.nelectron//2])] for x in mf.mo_energy],
}))
"#;

fn ctensor(v: &serde_json::Value) -> CTensor {
    let pull = |k: &str| -> Vec<f64> {
        v[k].as_array()
            .unwrap_or_else(|| panic!("payload has no {k}"))
            .iter()
            .map(|x| x.as_f64().expect("f64"))
            .collect()
    };
    CTensor {
        re: pull("re"),
        im: pull("im"),
    }
}

fn blocks(v: &serde_json::Value) -> Vec<CTensor> {
    v.as_array().expect("array").iter().map(ctensor).collect()
}

fn f64_blocks(v: &serde_json::Value) -> Vec<Vec<f64>> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|b| {
            b.as_array()
                .expect("block")
                .iter()
                .map(|x| x.as_f64().expect("f64"))
                .collect()
        })
        .collect()
}

fn max_abs_diff(a: &CTensor, b: &CTensor) -> f64 {
    assert_eq!(a.re.len(), b.re.len(), "block length mismatch");
    (0..a.re.len()).fold(0.0_f64, |w, i| {
        w.max((a.re[i] - b.re[i]).abs()).max((a.im[i] - b.im[i]).abs())
    })
}

/// Largest deviation, and how many compared values are BITWISE identical.
fn compare(got: &[Vec<f64>], want: &[Vec<f64>], label: &str) -> (f64, usize, usize) {
    assert_eq!(got.len(), want.len(), "{label}: block count");
    let (mut worst, mut total, mut bitwise) = (0.0_f64, 0usize, 0usize);
    for (b, (g_blk, w_blk)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(g_blk.len(), w_blk.len(), "{label}: block {b} orbital count");
        for (g, w) in g_blk.iter().zip(w_blk.iter()) {
            total += 1;
            if g.to_bits() == w.to_bits() {
                bitwise += 1;
            }
            worst = worst.max((g - w).abs());
        }
    }
    println!("{label}: {bitwise}/{total} bitwise identical, worst |delta| = {worst:e}");
    (worst, bitwise, total)
}

#[test]
#[ignore = "T1: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn krhf_get_bands_matches_upstream_on_yta() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let cell = yta_cscl();
    let nao = cell.mol.nao_nr;
    println!("YTa fixture: nao = {nao}, mesh = {:?}", cell.mesh);

    // Genuinely off the sampling mesh.
    let kband = cell
        .get_abs_kpts(&[[0.15, -0.07, 0.03], [-0.05, 0.11, 0.02]])
        .expect("scaled -> absolute k");

    let kpts = make_kpts_default(&cell, NK).expect("k-mesh");
    let df = Fftdf::new(cell.clone(), &kpts).expect("FFTDF");
    let mf = Krhf::from_df(Box::new(df));

    let kband_json: Vec<Vec<f64>> = kband.iter().map(|k| k.to_vec()).collect();
    let want = run_python(
        &py,
        BAND_PY,
        &cell_args(
            &cell,
            &[
                "gth-szv-molopt-sr".to_string(),
                "gth-pade".to_string(),
                serde_json::to_string(&NK.to_vec()).expect("json"),
                serde_json::to_string(&kband_json).expect("json"),
            ],
        ),
    );

    assert_eq!(
        want["version"].as_str(),
        Some("2.12.1"),
        "the oracle must be the VENDORED PySCF 2.12.1 — see tests/common/mod.rs"
    );
    assert!(
        want["converged"].as_bool().unwrap_or(false),
        "upstream did not converge"
    );

    // ---- 1. The same FFT mesh, derived independently on both sides. ----
    let mesh_ref: Vec<usize> = want["mesh"]
        .as_array()
        .expect("mesh")
        .iter()
        .map(|v| v.as_u64().expect("u64") as usize)
        .collect();
    assert_eq!(
        cell.mesh.to_vec(),
        mesh_ref,
        "mesh {:?} != upstream {mesh_ref:?} — different FFT grids",
        cell.mesh
    );

    // ---- 2. The same basis block and the same GTH q-channel. ----
    assert_eq!(nao as u64, want["nao"].as_u64().expect("nao"), "nao");
    assert_eq!(
        cell.mol.nelectron as u64,
        want["nelectron"].as_u64().expect("nelectron"),
        "electron count != upstream — one side resolved a different q-channel"
    );
    assert_eq!(cell.mol.nelectron, 24, "Y q11 + Ta q13 = 24 valence electrons");

    // ---- 3. The same cell. ----
    let e_nuc_ref = want["e_nuc"].as_f64().expect("e_nuc");
    assert!(
        (cell.energy_nuc().expect("e_nuc") - e_nuc_ref).abs() < 1e-10,
        "e_nuc != upstream {e_nuc_ref} — not the same cell"
    );

    // ---- 4. hcore: one-electron + GTH projectors, no density involved. ----
    let h_rust = mf.get_hcore().expect("rust get_hcore");
    let h_ref = blocks(&want["hcore"]);
    let worst_h = h_rust
        .iter()
        .zip(h_ref.iter())
        .enumerate()
        .map(|(k, (a, b))| {
            let d = max_abs_diff(a, b);
            println!("hcore k={k}: max |delta| = {d:e}");
            d
        })
        .fold(0.0_f64, f64::max);
    assert!(
        worst_h < 1e-10,
        "hcore disagrees by {worst_h:e} — the GTH q11/q13 projector path is wrong"
    );

    // ---- 5. The J/K path, on UPSTREAM'S converged density. ----
    //
    // This fixture is metallic: the k=0 HOMO/LUMO gap is ~2.9e-10 Ha, so two
    // independent aufbau SCFs pick different occupations and land 8.4e-2 Ha
    // apart. The density therefore comes from upstream.
    println!("upstream HOMO/LUMO per k: {}", want["homo_lumo"]);

    let mo_coeff = blocks(&want["mo_coeff_T"]);
    let mo_occ = f64_blocks(&want["mo_occ"]);
    let dm = vec![pyscf_pbc_scf::krdm::make_rdm1(&mo_coeff, &mo_occ, nao)];

    let vhf = mf.get_veff(&dm).expect("rust get_veff");
    // `.0` is ALREADY e1 + e_coul (khf.py:266) — do not add `.1` to it.
    let (e_elec, _) = mf.energy_elec(&dm, &h_rust, &vhf).expect("rust energy_elec");
    let e_tot_rust = e_elec + e_nuc_ref;
    let e_tot_ref = want["e_tot"].as_f64().expect("e_tot");
    println!(
        "e_tot on upstream's density: rust {e_tot_rust:.16} vs upstream \
         {e_tot_ref:.16} (delta {:e})",
        (e_tot_rust - e_tot_ref).abs()
    );
    assert!(
        (e_tot_rust - e_tot_ref).abs() < 1e-9,
        "this port's energy on UPSTREAM'S OWN density differs by {:e} Ha — a \
         real defect in the J/K path, not an SCF-solution difference",
        (e_tot_rust - e_tot_ref).abs()
    );

    // ---- 6. THE BAND GATE. ----
    let (e_band, _) = mf.get_bands(&kband, &dm).expect("get_bands");
    let e_band_v: Vec<Vec<f64>> = e_band.iter().map(|v| v.to_vec()).collect();
    let (w_band, bits, n) = compare(&e_band_v, &f64_blocks(&want["e_band"]), "get_bands (off mesh)");

    println!("BIT-EXACTNESS on Y/Ta: get_bands {bits}/{n} bitwise identical");
    assert!(
        w_band < BAND_TOL,
        "get_bands worst |delta| {w_band:e} exceeds {BAND_TOL:e}"
    );
}

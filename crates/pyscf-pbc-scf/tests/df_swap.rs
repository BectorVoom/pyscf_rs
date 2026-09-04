//! Plan 13-07 acceptance — every k-point driver takes any `PeriodicDf`
//! (D-PBC-22), and boxing the builder moved no number.
//!
//! Phase 11 hard-wired the concrete `Fftdf` into eight drivers and into
//! `pyscf_pbc_df::get_{nuc,pp,hcore}`. Until this plan landed, nothing
//! downstream could be handed an AFTDF — so **Phase 13's own Gate 2 was not
//! measurable**. These tests are the proof that it now is.

mod common;

use common::{diamond, he_all_electron};
use pyscf_pbc_df::traits::PeriodicDf;
use pyscf_pbc_df::{Aftdf, Fftdf};
use pyscf_pbc_scf::krhf::Krhf;
use pyscf_pbc_scf::types::KScfConfig;

fn cfg() -> KScfConfig {
    let mut c = KScfConfig::for_cell(&diamond());
    c.conv_tol = 1e-11;
    c.max_cycle = 60;
    c
}

/// **The refactor moved no number.** `Krhf::new` and an explicitly constructed
/// `Fftdf` must give BIT-IDENTICAL energies — this plan is a type change, not a
/// numerical one.
#[test]
fn boxing_the_builder_is_bit_identical() {
    let kpts = [[0.0; 3]];
    let a = Krhf::new(diamond(), &kpts).expect("krhf");
    let df = Fftdf::new(diamond(), &kpts).expect("fftdf");
    let b = Krhf::from_df(Box::new(df));
    let (ea, eb) = (
        a.kernel(&cfg()).expect("scf a").e_tot,
        b.kernel(&cfg()).expect("scf b").e_tot,
    );
    assert_eq!(
        ea.to_bits(),
        eb.to_bits(),
        "boxing the builder changed the energy: {ea:.17e} vs {eb:.17e}"
    );
}

/// The builder reports itself, for `dump_flags` and chkfile provenance.
#[test]
fn builders_name_themselves() {
    let kpts = [[0.0; 3]];
    assert_eq!(Fftdf::new(diamond(), &kpts).expect("fftdf").name(), "FFTDF");
    assert_eq!(Aftdf::new(diamond(), &kpts).expect("aftdf").name(), "AFTDF");
}

/// **`KRHF` runs on AFTDF — Gate 2, stated as the `(rcut, mesh)` ladder the
/// pre-implementation study measured.**
///
/// Diamond/`gth-szv`/`gth-pade`, the system every recorded number in
/// `.planning/phases/13-ft-ao-aftdf/measurements/` was taken on. The deviation
/// must FALL with the mesh and approach the recorded floor; it does not go to
/// zero, because FFTDF's pair density is aliased and AFTDF's is not.
///
/// Gamma point rather than 2×2×2 so this stays in a test suite's time budget —
/// the k-resolved rung is plan 13-08's, run once and recorded.
#[test]
fn gate2_krhf_aftdf_vs_fftdf_converges() {
    let kpts = [[0.0; 3]];
    let mut devs = Vec::new();
    for m in [15usize, 21, 27] {
        let mesh = [m, m, m];
        let f = Krhf::from_df(Box::new(
            Fftdf::with_mesh(diamond(), &kpts, mesh).expect("fftdf"),
        ));
        let a = Krhf::from_df(Box::new(
            Aftdf::with_mesh(diamond(), &kpts, mesh).expect("aftdf"),
        ));
        let rf = f.kernel(&cfg()).expect("fftdf scf");
        let ra = a.kernel(&cfg()).expect("aftdf scf");
        assert!(
            rf.converged && ra.converged,
            "both SCFs must converge at mesh {m}"
        );
        let d = (rf.e_tot - ra.e_tot).abs();
        eprintln!(
            "GATE2 mesh {m}: E_FFTDF {:.14} E_AFTDF {:.14} |dE| {:e}",
            rf.e_tot, ra.e_tot, d
        );
        devs.push(d);
    }
    assert!(devs[1] < devs[0], "mesh 21 must improve on mesh 15");
    assert!(devs[2] < devs[1], "mesh 27 must improve on mesh 21");
    assert!(
        devs[2] < 1e-7,
        "AFTDF and FFTDF still {:e} apart at mesh 27",
        devs[2]
    );
}

/// The all-electron path end to end — `get_nuc`, `_fake_nuc` and the
/// single-centre `ft_ao`, none of which a `gth-pade` cell reaches.
///
/// The tolerance is loose ON PURPOSE: He/`sto-3g` is a steep 1s Gaussian, so
/// FFTDF's aliasing at mesh 21 is ~3e-5 — three orders worse than smooth
/// diamond at the same mesh. That is FFTDF's error, not AFTDF's.
#[test]
fn krhf_runs_on_aftdf_all_electron() {
    let kpts = [[0.0; 3]];
    let mesh = [21usize, 21, 21];
    let f = Krhf::from_df(Box::new(
        Fftdf::with_mesh(he_all_electron(), &kpts, mesh).expect("fftdf"),
    ));
    let a = Krhf::from_df(Box::new(
        Aftdf::with_mesh(he_all_electron(), &kpts, mesh).expect("aftdf"),
    ));
    let mut c = KScfConfig::for_cell(&he_all_electron());
    c.conv_tol = 1e-11;
    c.max_cycle = 60;
    let rf = f.kernel(&c).expect("fftdf scf");
    let ra = a.kernel(&c).expect("aftdf scf");
    assert!(rf.converged && ra.converged, "both SCFs must converge");
    let d = (rf.e_tot - ra.e_tot).abs();
    eprintln!(
        "He gamma mesh 21: E_FFTDF {:.14} E_AFTDF {:.14} |dE| {:e}",
        rf.e_tot, ra.e_tot, d
    );
    assert!(d < 1e-4, "AFTDF and FFTDF disagree by {d:e} at mesh 21");
}

// ---------------------------------------------------------------------------
// Plan 14-04 — GDF drives the SCF. THE gate for Phase 14.
// ---------------------------------------------------------------------------

/// **GATE 1.** A converged `KRHF` on the ALL-ELECTRON He-fcc control, driven by
/// GDF, against upstream **−2.80842508664874**.
///
/// He-fcc is the control because `exclude_dd_block` is provably inert there
/// (D-PBC-23: `measurements/ddblock.py` measures its effect as exactly 0), so
/// the gate has no escape hatch.
///
/// The number exercises the WHOLE phase in one assertion: the auxiliary cell
/// (14-01), the fitting route (14-02 / 14-07), the store and the nuclear
/// builder (14-03), and the J/K contraction (14-04).
///
/// # BOTH routes are pinned, and that is the point
///
/// Plan 14-07 Task 7d flipped [`Gdf::prefer_ccdf`] to `false` on 2026-08-30, so
/// `Gdf::new` now takes the RANGE-SEPARATED route — upstream's default. The two
/// routes disagree by 5.222e-10 on this system, which is *inside* the 1e-9 bar
/// this test uses, so a version of this test that pinned only the compensated
/// number would have kept passing after the flip while silently measuring the
/// other route. That is exactly the drift Task 7d requires be made explicit, so
/// each route is now asserted against its OWN upstream number.
#[test]
fn krhf_on_gdf_matches_upstream_he_fcc() {
    use pyscf_pbc_df::gdf::Gdf;

    let cell = he_all_electron();
    let kpts =
        pyscf_pbc_gto::kpts_mesh::make_kpts(&cell, [2, 2, 2], false, true, None).expect("kpts");

    // `measurements/ccdf.py`, PySCF 2.12.1.
    const UPSTREAM_RS: f64 = -2.808_425_087_170_97; // `_prefer_ccdf = False`
    const UPSTREAM_CC: f64 = -2.808_425_086_648_74; // `_prefer_ccdf = True`

    for (route, prefer_ccdf, upstream) in [
        ("RS (default)", false, UPSTREAM_RS),
        ("CC", true, UPSTREAM_CC),
    ] {
        let mut c = KScfConfig::for_cell(&cell);
        c.conv_tol = 1e-11;
        c.max_cycle = 60;

        let mut df = Gdf::new(cell.clone(), &kpts);
        df.prefer_ccdf = prefer_ccdf;
        assert_eq!(
            Gdf::new(cell.clone(), &kpts).prefer_ccdf,
            false,
            "Task 7d: the DEFAULT route is the range-separated one"
        );
        let mf = Krhf::from_df(Box::new(df));
        let out = mf.kernel(&c).expect("KRHF on GDF");
        assert!(out.converged, "KRHF on GDF ({route}) did not converge");

        let d = (out.e_tot - upstream).abs();
        eprintln!(
            "KRHF/GDF {route} He-fcc 2x2x2: E = {:.14}, upstream {upstream:.14}, |dE| = {d:e}",
            out.e_tot
        );
        assert!(
            d < 1e-9,
            "KRHF on GDF ({route}): E = {:.14}, upstream {upstream:.14}, |dE| = {d:e}",
            out.e_tot
        );
    }
}

/// GDF is an APPROXIMATION and FFTDF is not, so their energies differ by the DF
/// fitting error — **6.006e-05 Ha** on He-fcc 2×2×2, measured upstream
/// (`measurements/builders.py`). This is the assertion that stops anyone
/// "fixing" the gap: it is a property of the auxiliary basis, present in
/// upstream, and reachable by no implementation.
#[test]
fn gdf_and_fftdf_differ_by_the_fitting_error() {
    use pyscf_pbc_df::gdf::Gdf;

    let cell = he_all_electron();
    let kpts =
        pyscf_pbc_gto::kpts_mesh::make_kpts(&cell, [2, 2, 2], false, true, None).expect("kpts");
    let mut c = KScfConfig::for_cell(&cell);
    c.conv_tol = 1e-11;
    c.max_cycle = 60;

    let e_gdf = Krhf::from_df(Box::new(Gdf::new(cell.clone(), &kpts)))
        .kernel(&c)
        .expect("gdf scf")
        .e_tot;
    let e_fft = Krhf::from_df(Box::new(
        Fftdf::new(he_all_electron(), &kpts).expect("fftdf"),
    ))
    .kernel(&c)
    .expect("fftdf scf")
    .e_tot;

    let d = (e_gdf - e_fft).abs();
    eprintln!("He-fcc |E_GDF - E_FFTDF| = {d:e} (upstream 6.006e-05)");
    assert!(
        (d - 6.006e-5).abs() < 1e-5,
        "the DF fitting error moved: {d:e}, upstream 6.006e-05"
    );
}

// ---------------------------------------------------------------------------
// Plan 14-06 — MDF. GATE 2.
// ---------------------------------------------------------------------------

fn krhf_e(df: Box<dyn PeriodicDf>, cell: &pyscf_pbc_gto::Cell) -> f64 {
    let mut c = KScfConfig::for_cell(cell);
    c.conv_tol = 1e-11;
    c.max_cycle = 60;
    let out = Krhf::from_df(df).kernel(&c).expect("SCF");
    assert!(out.converged, "SCF did not converge");
    out.e_tot
}

fn he_kpts(cell: &pyscf_pbc_gto::Cell, m: [usize; 3]) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::kpts_mesh::make_kpts(cell, m, false, true, None).expect("kpts")
}

/// **GATE 2 — MDF converges to FFTDF and GDF does not.** No oracle.
///
/// The target ladder is `measurements/mdfladder_cc.out`, which plan 14-06
/// added because **`mdfladder.out` measures the wrong builder**: it was
/// recorded with `df.MDF`'s default, and `MDF._prefer_ccdf` is `False`
/// (`mdf.py:79`), so every one of its rows is `_RSMDFBuilder` — plan 14-07's
/// route, not this one's. The CC ladder on He-fcc/`sto-3g` 2×2×2, against
/// `E_KRHF(FFTDF, mesh 31)`:
///
/// ```text
///   GDF (no plane waves)   6.002e-05
///   MDF mesh  7            1.695e-06
///   MDF mesh  9 (default)  5.476e-08
///   MDF mesh 11            6.684e-09   <- the plateau
///   MDF mesh 15            3.216e-08
/// ```
///
/// **The ladder is not monotone**, and a monotone gate would fail a correct
/// implementation: MDF's own auxiliary fit and the mesh-31 truncation of the
/// FFTDF *reference* are two independent floors, and past the crossover the
/// comparison measures the reference. Phase 13's Gate 2 had the same structure.
/// So the gate is: beat GDF by an order at mesh 7, fall two more orders by
/// mesh 11, beat GDF by three orders at the plateau, and stay within an order
/// of the plateau afterwards.
#[test]
fn gate2_mdf_converges_to_fftdf() {
    use pyscf_pbc_df::{Gdf, Mdf};

    let cell = he_all_electron();
    let kpts = he_kpts(&cell, [2, 2, 2]);

    let e_fft = krhf_e(
        Box::new(Fftdf::with_mesh(cell.clone(), &kpts, [31, 31, 31]).expect("fftdf")),
        &cell,
    );
    let e_gdf = krhf_e(Box::new(Gdf::new(cell.clone(), &kpts)), &cell);
    let d_gdf = (e_gdf - e_fft).abs();
    eprintln!("GATE2  GDF          |dE vs FFTDF| = {d_gdf:e}  (upstream 6.002e-05)");

    let mut devs = Vec::new();
    for m in [7usize, 11, 15] {
        let mut d = Mdf::new(cell.clone(), &kpts);
        d.mesh = Some([m, m, m]);
        let e = krhf_e(Box::new(d), &cell);
        let dev = (e - e_fft).abs();
        eprintln!("GATE2  MDF mesh {m:2}  E = {e:.14}  |dE vs FFTDF| = {dev:e}");
        devs.push(dev);
    }

    assert!(
        devs[0] < d_gdf / 10.0,
        "MDF at mesh 7 ({:e}) must already beat GDF ({d_gdf:e}) by an order",
        devs[0]
    );
    assert!(
        devs[1] < devs[0] / 100.0,
        "MDF must fall by >= 2 orders from mesh 7 ({:e}) to mesh 11 ({:e})",
        devs[0],
        devs[1]
    );
    assert!(
        devs[1] < d_gdf / 1000.0,
        "at the plateau MDF must beat GDF by >= 3 orders: {:e} vs {d_gdf:e}",
        devs[1]
    );
    assert!(
        devs[2] < devs[1] * 20.0,
        "past the plateau MDF must stay within an order of it: mesh 11 {:e}, \
         mesh 15 {:e}",
        devs[1],
        devs[2]
    );
}

/// **The structural check.** MDF is GDF plus a residual, so it must be
/// STRICTLY closer to the exact builder than GDF is. A sign error in
/// `add_ft_j3c` or in `mdf_jk`'s sum makes MDF diverge from *both* parents,
/// which is loud here and silent in a Hermiticity test.
#[test]
fn mdf_is_strictly_better_than_gdf() {
    use pyscf_pbc_df::{Gdf, Mdf};

    let cell = he_all_electron();
    let kpts = [[0.0; 3]];
    let e_fft = krhf_e(
        Box::new(Fftdf::with_mesh(cell.clone(), &kpts, [31, 31, 31]).expect("fftdf")),
        &cell,
    );
    let e_gdf = krhf_e(Box::new(Gdf::new(cell.clone(), &kpts)), &cell);
    let e_mdf = krhf_e(Box::new(Mdf::new(cell.clone(), &kpts)), &cell);
    let (dg, dm) = ((e_gdf - e_fft).abs(), (e_mdf - e_fft).abs());
    eprintln!("He gamma: |GDF - FFTDF| = {dg:e}, |MDF - FFTDF| = {dm:e}");
    assert!(
        dm < dg / 10.0,
        "MDF ({dm:e}) must beat GDF ({dg:e}) — if it does not, the plane-wave \
         residual is wired the wrong way round"
    );
}

/// Every builder drives `KRHF` through `Box<dyn PeriodicDf>` with no driver
/// change, and each names itself — the D-PBC-22 promise, extended to MDF.
#[test]
fn every_builder_drives_krhf_unchanged() {
    use pyscf_pbc_df::{Gdf, Mdf};

    let cell = he_all_electron();
    let kpts = [[0.0; 3]];
    let builders: Vec<(&str, Box<dyn PeriodicDf>)> = vec![
        (
            "FFTDF",
            Box::new(Fftdf::with_mesh(cell.clone(), &kpts, [31, 31, 31]).expect("fftdf")),
        ),
        (
            "AFTDF",
            Box::new(Aftdf::with_mesh(cell.clone(), &kpts, [31, 31, 31]).expect("aftdf")),
        ),
        ("GDF", Box::new(Gdf::new(cell.clone(), &kpts))),
        ("MDF", Box::new(Mdf::new(cell.clone(), &kpts))),
    ];
    for (name, df) in builders {
        assert_eq!(df.name(), name, "builder must name itself");
        let e = krhf_e(df, &cell);
        eprintln!("KRHF on {name}: E = {e:.14}");
        assert!(e.is_finite(), "{name} produced a non-finite energy");
    }
}

/// **The MDF oracle** — the converged `KRHF` energy against upstream
/// `mdf.MDF` with `_prefer_ccdf = True`, on the ALL-ELECTRON control.
///
/// The three upstream substitutions are the ones `pyscf-pbc-df`'s
/// `tests/df_ao2mo.rs` documents and measures: `exclude_dd_block = False`
/// (D-PBC-23, provably inert on He-fcc), `direct_scf_tol = 1e-14`, and a
/// UNIFORM `estimate_rcut` in place of `ft_ao.ExtendedMole.strip_basis`'s
/// per-shell-pair radii.
///
/// **The third one matters far more here than it does for GDF, and that is a
/// finding of this plan.** MDF's metric is `<g|g> - <g|G><G|g>` — the Gaussian
/// overlap with the plane-wave projection removed — and it is deliberately
/// near-singular: measured on He-fcc at gamma, its smallest RETAINED
/// eigenvalue is **2.464e-08** against a largest of 1.168, so `solve_cderi`'s
/// pseudo-inverse amplifies any error in `j3c` by up to **4.1e7**. The
/// `strip_basis` residual this port carries is 1.22e-09 in `j3c`; through that
/// amplification it is worth ~4.7e-05 on the converged energy, which is why
/// `mdf_is_strictly_better_than_gdf` cannot pass on the DEFAULT upstream route.
/// Upstream says the same thing in its own words at `mdf.py:362-365` — "small
/// integral errors can lead to a difference in the total energy […] around 4th
/// decimal place" — which is why it abandons Cholesky for MDF.
#[test]
#[ignore = "oracle: needs PYSCF_ORACLE_VENV and a real MDF build"]
fn krhf_on_mdf_matches_upstream_he_fcc() {
    use pyscf_pbc_df::Mdf;

    let Some(py) = common::oracle_python() else {
        eprintln!("{} unset — skipping", common::GATE);
        return;
    };
    let cell = he_all_electron();
    for (kmesh, mesh) in [([1usize, 1, 1], [11usize, 11, 11]), ([2, 2, 2], [9, 9, 9])] {
        let kpts = he_kpts(&cell, kmesh);
        let mut d = Mdf::new(cell.clone(), &kpts);
        d.mesh = Some(mesh);
        let got = krhf_e(Box::new(d), &cell);

        let v = common::run_python(&py, MDF_ORACLE, &mdf_oracle_args(&cell, kmesh, mesh));
        assert_eq!(
            v["converged"].as_bool(),
            Some(true),
            "upstream must converge"
        );
        let want = v["e_tot"].as_f64().expect("e_tot");
        let dev = (got - want).abs();
        eprintln!(
            "KRHF/MDF He-fcc {kmesh:?} mesh {mesh:?}: port {got:.14}, upstream {want:.14}, \
             |dE| = {dev:e}"
        );
        // 1e-8, not the 1e-11 the GDF gates carry, and the reason is measured
        // rather than assumed: MDF's metric is deliberately near-singular
        // (smallest retained eigenvalue 2.464e-08 against a largest of 1.168 on
        // He-fcc at gamma), so `solve_cderi`'s pseudo-inverse amplifies any
        // residual in `j3c` by up to 4.1e7. With the screens equalised the port
        // reaches 3.601e-09 at gamma — four orders inside upstream's own stated
        // sensitivity for this builder ("around 4th decimal place",
        // `mdf.py:362-365`).
        assert!(
            dev < 1e-8,
            "KRHF on MDF vs upstream (screens equalised) at {kmesh:?}: |dE| = {dev:e}"
        );
    }
}

fn mdf_oracle_args(cell: &pyscf_pbc_gto::Cell, kmesh: [usize; 3], mesh: [usize; 3]) -> Vec<String> {
    let a: Vec<Vec<f64>> = cell.a.iter().map(|r| r.to_vec()).collect();
    let xyz: Vec<Vec<f64>> = cell.mol.atom_coords().iter().map(|r| r.to_vec()).collect();
    let sym: Vec<String> = cell.mol._atom.iter().map(|(s, _)| s.clone()).collect();
    vec![
        serde_json::to_string(&a).expect("json"),
        serde_json::to_string(&xyz).expect("json"),
        serde_json::to_string(&sym).expect("json"),
        serde_json::to_string(&kmesh.to_vec()).expect("json"),
        serde_json::to_string(&mesh.to_vec()).expect("json"),
    ]
}

const MDF_ORACLE: &str = r#"
import json, sys
import numpy
import pyscf
assert pyscf.__version__ == '2.12.1', pyscf.__version__
from pyscf.pbc import gto as pgto, scf as pscf
from pyscf.pbc.df import mdf
import pyscf.pbc.df.gdf_builder as gb

a, xyz, sym, kmesh, mesh = (
    json.loads(sys.argv[1]), json.loads(sys.argv[2]), json.loads(sys.argv[3]),
    json.loads(sys.argv[4]), json.loads(sys.argv[5]))

cell = pgto.Cell()
cell.a = numpy.array(a)
cell.atom = [(s, tuple(c)) for s, c in zip(sym, xyz)]
cell.basis = 'sto-3g'
cell.unit = 'Bohr'
cell.verbose = 0
cell.build()
kpts = cell.make_kpts(kmesh)

_init = gb._CCGDFBuilder.__init__
def _patched_init(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    self.exclude_dd_block = False
gb._CCGDFBuilder.__init__ = _patched_init

_build = gb._CCGDFBuilder.build
def _patched_build(self, *args, **kwargs):
    out = _build(self, *args, **kwargs)
    self.direct_scf_tol = 1e-14
    return out
gb._CCGDFBuilder.build = _patched_build

# ExtendedMole.strip_basis, defeated: a per-shell-pair radius array flattened
# to its own maximum keeps every image this port keeps.
_erc = gb.estimate_rcut
def _uniform_rcut(*args, **kwargs):
    r = numpy.asarray(_erc(*args, **kwargs), dtype=float)
    return numpy.full_like(r, r.max())
gb.estimate_rcut = _uniform_rcut

d = mdf.MDF(cell, kpts)
d._prefer_ccdf = True
d.mesh = mesh
mf = pscf.KRHF(cell, kpts)
mf.with_df = d
mf.exxdiv = 'ewald'
mf.conv_tol = 1e-11
e = mf.kernel()
print(json.dumps({'e_tot': float(e), 'mesh': [int(x) for x in d.mesh],
                  'converged': bool(mf.converged)}))
"#;

/// **Wall clock, per builder** — plan 14-09 Task 2. `#[ignore]`d because it is
/// a measurement, not an assertion.
///
/// Upstream's numbers on diamond `gth-szv` 2×2×2 (`measurements/builders.out`):
/// GDF **6.4 s**, RSDF 13.5 s, MDF 16.9 s, FFTDF 30.0 s, AFTDF 450.6 s. **GDF
/// being the FASTEST builder is the phase's whole point**; if this port inverts
/// that ordering, the number belongs in `14-VERIFICATION.md` with the reason.
///
/// Measured here on He-fcc/`sto-3g` 2×2×2 rather than diamond, because one
/// diamond `make_j3c` is a single screening group of ~77 M cintx shell triples
/// and its wall time is still unmeasured (§3 of the verification).
#[test]
#[ignore = "measurement: prints wall clock, asserts nothing"]
fn wall_clock_per_builder() {
    use pyscf_pbc_df::{Gdf, Mdf};
    use std::time::Instant;

    let cell = he_all_electron();
    let kpts = he_kpts(&cell, [2, 2, 2]);

    let mut rows: Vec<(String, f64, f64)> = Vec::new();
    for name in ["FFTDF", "AFTDF", "GDF", "MDF"] {
        let t0 = Instant::now();
        let mut df: Box<dyn PeriodicDf> = match name {
            "FFTDF" => {
                Box::new(Fftdf::with_mesh(cell.clone(), &kpts, [31, 31, 31]).expect("fftdf"))
            }
            "AFTDF" => {
                Box::new(Aftdf::with_mesh(cell.clone(), &kpts, [31, 31, 31]).expect("aftdf"))
            }
            "GDF" => Box::new(Gdf::new(cell.clone(), &kpts)),
            _ => Box::new(Mdf::new(cell.clone(), &kpts)),
        };
        df.build().expect("build");
        let build = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let e = krhf_e(df, &cell);
        rows.push((name.to_string(), build, t1.elapsed().as_secs_f64()));
        eprintln!(
            "WALL {name:6}  build {:7.2} s  scf {:7.2} s  E = {e:.14}",
            rows[rows.len() - 1].1,
            rows[rows.len() - 1].2
        );
    }
    eprintln!(
        "WALL (He-fcc/sto-3g 2x2x2; upstream's diamond 2x2x2 reference: GDF 6.4 s, MDF 16.9 s, FFTDF 30.0 s, AFTDF 450.6 s)"
    );
}

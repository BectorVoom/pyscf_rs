//! Plan 20-04 — bisect the one unmet numeric gate by ROUTE STEP.
//!
//! `17-VERIFICATION.md:126`: `KRKS` k-symmetry vs full BZ on **GDF** measured
//! `1.432e-06` (`krks_ksymm.rs::krks_ibz_energy_matches_full_bz_on_gdf`),
//! while the FFTDF arm of the identical comparison passes at `3.109e-14`.
//!
//! Every test here is a measurement, not a regression gate: it prints
//! `max |Δ|` per step and asserts only the NON-VACUITY preconditions (the
//! fixture folds, the band set is a strict bitwise subset, the symmetrised
//! density round-trips). The findings are recorded in
//! `.planning/phases/20-pbc-python-bindings/measurements/gdf-ksymm-bisect.md`.
//!
//! The two arms are built EXACTLY as the failing gate builds them:
//!
//! * full BZ: `Krks::from_df(Box::new(Gdf::new(cell, kpts)), "lda,vwn")`;
//! * IBZ: `KsymAdaptedKrks::from_df(Box::new(Gdf::new(cell, kpts)), kpts,
//!   "lda,vwn", PeriodicGrids::uniform(&cell, Some(cell.mesh)))`.
//!
//! and every step is evaluated on ONE density, so nothing measures SCF drift.

use pyscf_algebra::CTensor;
use pyscf_pbc_df::{Fftdf, Gdf, JkOpts, PeriodicDf, get_hcore};
use pyscf_pbc_dft::gen_grid::PeriodicGrids;
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_dft::krks_ksymm::KsymAdaptedKrks;
use pyscf_pbc_dft::numint::unfold_kdms_sym;
use pyscf_pbc_dft::veff::get_jk;
use pyscf_pbc_gto::test_systems::si;
use pyscf_pbc_gto::{Cell, ExxDiv, make_kpts_default};
use pyscf_pbc_scf::khooks::KOverrideHooks;
use pyscf_pbc_scf::krdm::trace_ab;
use pyscf_pbc_scf::{KInitGuess, KScfConfig};
use pyscf_pbc_symm::basis::{self, SymmAdaptedBasisInput};
use pyscf_pbc_symm::kpts::{KPoints, make_kpts};

const XC: &str = "lda,vwn";
/// As `krks_ksymm.rs` — D-17-07-01.
const TIME_REVERSAL: bool = false;
/// The plan's bisect threshold.
const STEP_TOL: f64 = 1e-9;

/// The gate's own fixture: `si()` at DEFAULT precision, `[2,2,2]`, symmetry
/// basis built. Task 1's non-vacuity assertions live here.
fn fixture() -> (Cell, KPoints) {
    let mut cell = si();
    let kpts_abs = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts_default");
    let kpts = make_kpts(&cell, &kpts_abs, true, TIME_REVERSAL).expect("make_kpts");
    let (nk, nibz) = (kpts.nkpts(), kpts.nkpts_ibz());
    println!("[fixture] nkpts = {nk}, nkpts_ibz = {nibz}");
    assert!(
        nibz < nk,
        "the fixture must fold: nkpts_ibz = {nibz}, nkpts = {nk}"
    );
    // The band set the ksymm driver hands `get_jk` is `kpts_ibz`. It must be a
    // STRICT subset of the sampling set, and bitwise so (memory:
    // ksymm-ibz-kpts-were-not-a-bitwise-subset), or `band_is_kpts` would
    // short-circuit / the union rebuild would grow the k-set.
    assert_eq!(kpts.kpts_ibz.len(), nibz);
    assert!(
        kpts.kpts_ibz.len() < kpts.kpts.len(),
        "band set must be a STRICT subset"
    );
    for (i, kb) in kpts.kpts_ibz.iter().enumerate() {
        let k = kpts.kpts[kpts.ibz2bz[i]];
        assert!(
            kb.iter()
                .zip(k.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "kpts_ibz[{i}] is not bitwise kpts[ibz2bz[{i}]]"
        );
    }
    println!(
        "[fixture] ibz2bz = {:?}, weights_ibz = {:?}",
        kpts.ibz2bz, kpts.weights_ibz
    );
    let input = SymmAdaptedBasisInput {
        kpts_scaled_ibz: kpts.kpts_scaled_ibz.clone(),
        little_cogroup_ops: kpts.little_cogroup_ops.clone(),
        ops: kpts.symmetry.ops.clone(),
        dmats: kpts.symmetry.dmats.clone(),
    };
    basis::build_symmetry(&mut cell, &input).expect("build_symmetry");
    (cell, kpts)
}

fn scf_cfg() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-10,
        conv_tol_grad: Some(1e-8),
        max_cycle: 50,
        init_guess: KInitGuess::Minao,
        ..KScfConfig::default()
    }
}

fn grid_mesh(g: &PeriodicGrids) -> String {
    match g {
        PeriodicGrids::Uniform(u) => format!("Uniform{:?} ({} pts)", u.mesh, u.coords.len()),
        PeriodicGrids::Becke(_) => "Becke".to_string(),
    }
}

fn max_abs(a: &CTensor, b: &CTensor) -> f64 {
    assert_eq!(a.re.len(), b.re.len(), "shape mismatch");
    let mut w = 0.0_f64;
    for i in 0..a.re.len() {
        w = w.max((a.re[i] - b.re[i]).abs());
        w = w.max((a.im[i] - b.im[i]).abs());
    }
    w
}

/// `full[ibz2bz[j]]` vs `ibz[j]`.
fn max_at_ibz(full: &[CTensor], ibz: &[CTensor], kpts: &KPoints) -> f64 {
    assert_eq!(ibz.len(), kpts.nkpts_ibz());
    ibz.iter()
        .enumerate()
        .map(|(j, m)| max_abs(&full[kpts.ibz2bz[j]], m))
        .fold(0.0, f64::max)
}

fn verdict(x: f64) -> &'static str {
    if x > STEP_TOL { "EXCEEDS 1e-9" } else { "ok" }
}

/// One converged full-BZ density, symmetrised so that BOTH arms see the SAME
/// full-BZ density bit for bit: `D_sym = unfold(D[ibz2bz])`, `D_ibz =
/// D[ibz2bz]`, and `unfold(D_sym[ibz2bz]) == D_sym` is asserted bitwise.
fn one_density(cell: &Cell, kpts: &KPoints) -> (CTensorStack, CTensorStack) {
    let nao = cell.mol.nao_nr;
    let mf = Krks::new(cell.clone(), &kpts.kpts, XC).expect("Krks");
    let r = mf.kernel(&scf_cfg()).expect("FFTDF full-BZ KRKS");
    assert!(r.converged, "density SCF did not converge");
    println!("[density] FFTDF full-BZ KRKS e_tot = {:.12}", r.e_tot);
    let d = &r.dm[0];
    let d_ibz: Vec<CTensor> = kpts.ibz2bz.iter().map(|&k| d[k].clone()).collect();
    let d_sym = unfold_kdms_sym(kpts, cell, &vec![d_ibz.clone()], nao)
        .expect("unfold")
        .into_owned()
        .remove(0);
    let brk = d
        .iter()
        .zip(d_sym.iter())
        .map(|(a, b)| max_abs(a, b))
        .fold(0.0, f64::max);
    println!(
        "[density] symmetry breaking of the converged density max|D - unfold(D_ibz)| = {brk:e}"
    );
    // Round trip: D_sym at the IBZ points is D_ibz, so unfolding it again is a
    // bitwise no-op. If not, the two arms would not see one density.
    let d_ibz2: Vec<CTensor> = kpts.ibz2bz.iter().map(|&k| d_sym[k].clone()).collect();
    let rt = max_at_ibz(&d_sym, &d_ibz, kpts);
    let again = unfold_kdms_sym(kpts, cell, &vec![d_ibz2], nao)
        .expect("unfold")
        .into_owned()
        .remove(0);
    let rt2 = d_sym
        .iter()
        .zip(again.iter())
        .map(|(a, b)| max_abs(a, b))
        .fold(0.0, f64::max);
    println!("[density] D_sym[ibz2bz] vs D_ibz = {rt:e}; unfold round trip = {rt2:e}");
    assert_eq!(rt, 0.0, "D_sym must equal D at the IBZ points");
    assert_eq!(
        rt2, 0.0,
        "the unfold must be a bitwise fixed point on D_sym"
    );
    (d_sym, d_ibz)
}

type CTensorStack = Vec<CTensor>;

fn jk_opts<'a>(band: Option<&'a [[f64; 3]]>, with_k: bool) -> JkOpts<'a> {
    JkOpts {
        hermi: 1,
        kpts_band: band,
        with_j: true,
        with_k,
        exxdiv: Some(ExxDiv::Ewald),
        omega: None,
        kk_symmetry: false,
    }
}

/// Steps 2-7 on a pair of arms, whatever the DF route. Returns the first step
/// name exceeding `STEP_TOL` (or `None`).
#[allow(clippy::too_many_lines)]
fn bisect_arms(
    route: &str,
    cell: &Cell,
    kpts: &KPoints,
    full: &mut Krks,
    ibz: &KsymAdaptedKrks,
    d_sym: &[CTensor],
    d_ibz: &[CTensor],
    with_k: bool,
) -> Option<String> {
    let nao = cell.mol.nao_nr;
    let band = kpts.kpts_ibz.clone();
    let nk = kpts.nkpts() as f64;
    let dms_full = vec![d_sym.to_vec()];
    let mut first: Option<String> = None;
    let mut record = |name: &str, x: f64| {
        println!("[{route}] {name:<58} max|Δ| = {x:e}  {}", verdict(x));
        if x > STEP_TOL && first.is_none() {
            first = Some(name.to_string());
        }
    };

    println!(
        "[{route}] cell.mesh = {:?}; with_df.mesh() full = {:?}, ibz = {:?}",
        cell.mesh,
        full.with_df.mesh(),
        ibz.with_df.mesh()
    );
    println!(
        "[{route}] XC grid: full arm = {}, ibz arm = {}",
        grid_mesh(&full.grids),
        grid_mesh(&ibz.grids)
    );

    // Step 2 — J: direct on the full arm's DF vs the ksymm path (band = IBZ)
    // on the ksymm arm's DF, the exact `veff::get_jk` both drivers call.
    let j_full = get_jk(
        full.with_df.as_ref(),
        XC,
        &dms_full,
        1,
        &kpts.kpts,
        None,
        Some(ExxDiv::Ewald),
        true,
    )
    .expect("full J")
    .vj
    .expect("vj");
    let j_ibz = get_jk(
        ibz.with_df.as_ref(),
        XC,
        &dms_full,
        1,
        &kpts.kpts,
        Some(&band),
        Some(ExxDiv::Ewald),
        true,
    )
    .expect("ibz J")
    .vj
    .expect("vj");
    record(
        "step 2  get_j  full-BZ direct vs IBZ (kpts_band) path",
        max_at_ibz(&j_full[0], &j_ibz[0], kpts),
    );
    let ecoul_full = 0.5 / nk
        * (0..kpts.nkpts())
            .map(|k| trace_ab(&d_sym[k], &j_full[0][k], nao).0)
            .sum::<f64>();
    let ecoul_ibz = 0.5
        * (0..kpts.nkpts_ibz())
            .map(|j| kpts.weights_ibz[j] * trace_ab(&d_ibz[j], &j_ibz[0][j], nao).0)
            .sum::<f64>();
    record(
        "step 2e ecoul  1/nk trace vs weights_ibz trace",
        (ecoul_full - ecoul_ibz).abs(),
    );

    // Step 3 — K (not used by LDA, measured for completeness).
    if with_k {
        let k_full = full
            .with_df
            .get_jk(&dms_full, &kpts.kpts, jk_opts(None, true))
            .expect("full K");
        let k_ibz = ibz
            .with_df
            .get_jk(&dms_full, &kpts.kpts, jk_opts(Some(&band), true))
            .expect("ibz K");
        record(
            "step 3  get_k  full-BZ direct vs IBZ (kpts_band) path",
            max_at_ibz(
                &k_full.vk.as_ref().expect("vk")[0],
                &k_ibz.vk.as_ref().expect("vk")[0],
                kpts,
            ),
        );
        // Step 4 — the band route vs the direct route on ONE DF object.
        let k_dir = ibz
            .with_df
            .get_jk(&dms_full, &kpts.kpts, jk_opts(None, true))
            .expect("direct");
        record(
            "step 4  kpts_band route vs direct, same DF object: vj",
            max_at_ibz(
                &k_dir.vj.as_ref().expect("vj")[0],
                &k_ibz.vj.as_ref().expect("vj")[0],
                kpts,
            ),
        );
        record(
            "step 4  kpts_band route vs direct, same DF object: vk",
            max_at_ibz(
                &k_dir.vk.as_ref().expect("vk")[0],
                &k_ibz.vk.as_ref().expect("vk")[0],
                kpts,
            ),
        );
    }

    // Step 5 — hcore (`get_pp` + `T`) at the IBZ points vs the full set.
    let h_full = get_hcore(full.with_df.as_ref(), &kpts.kpts).expect("hcore full");
    let h_ibz = get_hcore(ibz.with_df.as_ref(), &kpts.kpts_ibz).expect("hcore ibz");
    record(
        "step 5  hcore  full set vs IBZ set",
        max_at_ibz(&h_full, &h_ibz, kpts),
    );

    // Step 6 — XC, each arm on ITS OWN grid (as the gate runs them).
    let x_full = full
        .ni
        .nr_rks(cell, &full.grids, XC, &dms_full, 1, &kpts.kpts, None)
        .expect("xc full");
    let x_ibz = ibz
        .ni
        .nr_rks(cell, &ibz.grids, XC, &dms_full, 1, &kpts.kpts, Some(&band))
        .expect("xc ibz");
    println!(
        "[{route}] nelec full = {:.12}, ibz = {:.12}; exc full = {:.12}, ibz = {:.12}",
        x_full.nelec, x_ibz.nelec, x_full.exc, x_ibz.exc
    );
    record(
        "step 6  vxc    each arm on its own XC grid",
        max_at_ibz(&x_full.vmat[0], &x_ibz.vmat[0], kpts),
    );
    record(
        "step 6e exc    each arm on its own XC grid",
        (x_full.exc - x_ibz.exc).abs(),
    );

    // Step 7 — the production energy functional of both drivers at this one
    // density (`get_veff` then `energy_elec`, the SCF's own calls).
    let e_of = |mf: &dyn KOverrideHooks, dms: &Vec<Vec<CTensor>>| -> f64 {
        let h = mf.get_hcore().expect("hcore");
        let v = mf.get_veff(dms).expect("veff");
        mf.energy_elec(dms, &h, &v).expect("energy_elec").0
    };
    let e_full = e_of(full, &dms_full);
    let e_ibz = e_of(ibz, &vec![d_ibz.to_vec()]);
    println!("[{route}] E_elec[D] full = {e_full:.12}, ibz = {e_ibz:.12}");
    record(
        "step 7  E_elec[D] production drivers, as the gate builds them",
        (e_full - e_ibz).abs(),
    );

    // Step 8 — the same two with the full arm's XC grid MATCHED to the ksymm
    // arm's (`cell.mesh`, upstream's `UniformGrids(cell)`).
    full.grids = PeriodicGrids::uniform(cell, Some(cell.mesh)).expect("grid");
    let x_full2 = full
        .ni
        .nr_rks(cell, &full.grids, XC, &dms_full, 1, &kpts.kpts, None)
        .expect("xc full matched");
    println!(
        "[{route}] matched grid: full arm = {}",
        grid_mesh(&full.grids)
    );
    record(
        "step 8  vxc    full arm grid matched to cell.mesh",
        max_at_ibz(&x_full2.vmat[0], &x_ibz.vmat[0], kpts),
    );
    record(
        "step 8e exc    full arm grid matched to cell.mesh",
        (x_full2.exc - x_ibz.exc).abs(),
    );
    let e_full2 = e_of(full, &dms_full);
    println!("[{route}] E_elec[D] full (matched grid) = {e_full2:.12}, ibz = {e_ibz:.12}");
    record(
        "step 8E E_elec[D] full arm grid matched to cell.mesh",
        (e_full2 - e_ibz).abs(),
    );

    println!("[{route}] FIRST STEP > 1e-9: {first:?}");
    first
}

/// Task 1 in isolation, and the cheapest discriminating fact: which XC grid
/// each arm of the failing gate is handed. No 3-centre build.
#[test]
#[ignore = "T1: 20-04 bisect diagnostic -- fixture non-vacuity and per-arm XC grid, no GDF build"]
fn bisect_fixture_and_arm_grids() {
    let (cell, kpts) = fixture();
    let gdf = Gdf::new(cell.clone(), &kpts.kpts);
    let fft = Fftdf::new(cell.clone(), &kpts.kpts).expect("Fftdf");
    println!(
        "[grids] cell.mesh = {:?}; Gdf::mesh() = {:?}; Fftdf::mesh() = {:?}",
        cell.mesh,
        gdf.mesh(),
        fft.mesh()
    );
    let full_gdf = Krks::from_df(Box::new(gdf), XC).expect("Krks gdf");
    let full_fft = Krks::from_df(Box::new(fft), XC).expect("Krks fft");
    let ibz_grid = PeriodicGrids::uniform(&cell, Some(cell.mesh)).expect("grid");
    println!(
        "[grids] Krks::from_df(Gdf)   XC grid = {}",
        grid_mesh(&full_gdf.grids)
    );
    println!(
        "[grids] Krks::from_df(Fftdf) XC grid = {}",
        grid_mesh(&full_fft.grids)
    );
    println!(
        "[grids] KsymAdaptedKrks (gate)  XC grid = {}",
        grid_mesh(&ibz_grid)
    );
}

/// Task 2 + Task 3: every step, GDF and FFTDF, from ONE density.
#[test]
#[ignore = "T3: 20-04 bisect diagnostic -- GDF _cderi/J/K/band/XC/energy steps plus FFTDF control, minutes-scale"]
fn bisect_gdf_and_fftdf_routes_at_one_density() {
    let (cell, kpts) = fixture();
    let (d_sym, d_ibz) = one_density(&cell, &kpts);

    // ---- GDF --------------------------------------------------------------
    // Step 1 — `_cderi`. Both arms build over the FULL BZ (the ksymm driver
    // never builds an IBZ fit, D-PBC-15); compare the two independent builds
    // block by block, then compare against a fit built over the IBZ k-set
    // alone at its diagonal pairs.
    let gdf_full = Gdf::new(cell.clone(), &kpts.kpts);
    let gdf_ibzarm = Gdf::new(cell.clone(), &kpts.kpts);
    let c_full = gdf_full.cderi().expect("cderi full");
    let c_ibzarm = gdf_ibzarm.cderi().expect("cderi ibz arm");
    assert_eq!(c_full.blocks.len(), c_ibzarm.blocks.len());
    let mut w1 = 0.0_f64;
    for (key, b) in &c_full.blocks {
        let o = &c_ibzarm.blocks[key];
        assert_eq!(
            (b.rank, b.nao_pair),
            (o.rank, o.nao_pair),
            "block {key} shape"
        );
        w1 = w1.max(max_abs(&b.data, &o.data));
    }
    println!(
        "[GDF] step 1a _cderi two full-BZ builds ({} blocks)            max|Δ| = {w1:e}  {}",
        c_full.blocks.len(),
        verdict(w1)
    );
    {
        let gdf_ibzonly = Gdf::new(cell.clone(), &kpts.kpts_ibz);
        let c_i = gdf_ibzonly.cderi().expect("cderi ibz-only");
        let mut w = 0.0_f64;
        let mut shapes = Vec::new();
        for j in 0..kpts.nkpts_ibz() {
            let k = kpts.ibz2bz[j];
            let a = c_full.get(k, k).expect("full (k,k)");
            let b = c_i.get(j, j).expect("ibz (j,j)");
            shapes.push(((a.rank, a.nao_pair), (b.rank, b.nao_pair)));
            if (a.rank, a.nao_pair) == (b.rank, b.nao_pair) {
                w = w.max(max_abs(&a.data, &b.data));
            } else {
                w = f64::INFINITY;
            }
        }
        println!(
            "[GDF] step 1b _cderi (k,k) blocks: full-BZ fit vs IBZ-only fit  max|Δ| = {w:e}  {} shapes {shapes:?}",
            verdict(w)
        );
    }

    let mut full = Krks::from_df(Box::new(gdf_full), XC).expect("Krks gdf");
    let ibz = KsymAdaptedKrks::from_df(
        Box::new(gdf_ibzarm),
        kpts.clone(),
        XC,
        PeriodicGrids::uniform(&cell, Some(cell.mesh)).expect("grid"),
    );
    let first_gdf = bisect_arms("GDF", &cell, &kpts, &mut full, &ibz, &d_sym, &d_ibz, true);

    // ---- FFTDF control ----------------------------------------------------
    let mut full_f = Krks::new(cell.clone(), &kpts.kpts, XC).expect("Krks fft");
    let ibz_f = KsymAdaptedKrks::new(cell.clone(), kpts.clone(), XC).expect("ksymm fft");
    let first_fft = bisect_arms(
        "FFTDF",
        &cell,
        &kpts,
        &mut full_f,
        &ibz_f,
        &d_sym,
        &d_ibz,
        true,
    );

    println!(
        "[summary] GDF first step > 1e-9: {first_gdf:?}; FFTDF first step > 1e-9: {first_fft:?}"
    );
}

/// Confirmation at the gate's own level: the failing gate re-run as written,
/// and again with ONLY the full-BZ arm's XC grid matched to `cell.mesh`.
#[test]
#[ignore = "T3: 20-04 bisect diagnostic -- three GDF KRKS SCFs (~25-35 min), confirms the named step"]
fn gdf_gate_c_with_and_without_matched_xc_grid() {
    let (cell, kpts) = fixture();
    let cfg = scf_cfg();
    let grid = || PeriodicGrids::uniform(&cell, Some(cell.mesh)).expect("grid");

    let ibz = KsymAdaptedKrks::from_df(
        Box::new(Gdf::new(cell.clone(), &kpts.kpts)),
        kpts.clone(),
        XC,
        grid(),
    );
    let r_ibz = ibz.kernel(&cfg).expect("IBZ GDF KRKS");
    assert!(r_ibz.converged);
    println!(
        "[gate] e_ibz = {:.12} (XC grid {})",
        r_ibz.e_tot,
        grid_mesh(&ibz.grids)
    );

    let full = Krks::from_df(Box::new(Gdf::new(cell.clone(), &kpts.kpts)), XC).expect("Krks");
    let r_full = full.kernel(&cfg).expect("full GDF KRKS");
    assert!(r_full.converged);
    let de = (r_full.e_tot - r_ibz.e_tot).abs();
    println!(
        "[gate] AS WRITTEN: e_full = {:.12} (XC grid {}), |dE| = {de:e}",
        r_full.e_tot,
        grid_mesh(&full.grids)
    );

    let mut full2 = Krks::from_df(Box::new(Gdf::new(cell.clone(), &kpts.kpts)), XC).expect("Krks");
    full2.grids = grid();
    let r_full2 = full2.kernel(&cfg).expect("full GDF KRKS matched");
    assert!(r_full2.converged);
    let de2 = (r_full2.e_tot - r_ibz.e_tot).abs();
    println!(
        "[gate] MATCHED GRID: e_full = {:.12} (XC grid {}), |dE| = {de2:e}",
        r_full2.e_tot,
        grid_mesh(&full2.grids)
    );
}

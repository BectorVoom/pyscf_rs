//! W-00 (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`) — the committed,
//! re-runnable KRKS SCF profiling harness. Replaces the throwaway pair of
//! examples §2.1 of that plan was measured with (deleted after that run).
//!
//! ```bash
//! # Isolated 3-D transform batch — the ~93% of get_k_kpts that W-02 targets.
//! cargo run -p pyscf-bench --release --bin krks_profile -- transform
//!
//! # get_j_kpts / get_k_kpts / a full SCF, for one cell/k-mesh/FFT-mesh/xc.
//! cargo run -p pyscf-bench --release --bin krks_profile -- \
//!     jk --cell si --nk 2,2,2 --mesh 31,31,31 --xc pbe0 --json out.json
//!
//! # Diff a run against a committed baseline.
//! cargo run -p pyscf-bench --release --bin krks_profile -- \
//!     jk --cell si --nk 2,2,2 --mesh 31,31,31 --xc pbe0 --compare baseline.json
//! ```
//!
//! Every stage is timed WARM (after one throwaway pass, per the plan's RULE
//! O and `11_launch_overhead_and_transfers.md` §6) so the AO cache, the
//! `coulG`/`expmikr` cache (W-01) and the FFT plan cache are all paid for
//! before the reported number.

use std::time::Instant;

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_df::{Fftdf, get_hcore, get_j_kpts, get_k_kpts};
use pyscf_pbc_dft::krks::{Krks, hybrid};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_gto::hcore::get_ovlp;
use pyscf_pbc_scf::KScfConfig;
use pyscf_pbc_tools::coulg::ExxDiv;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Cells — same geometries as `pyscf-pbc-dft/tests/common/mod.rs` (not a
// public API of that crate, so reproduced here rather than depended on).
// ---------------------------------------------------------------------------

fn bohr_cell(a: [[f64; 3]; 3], atoms: Vec<(String, [f64; 3])>, pseudo: Option<&str>) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(a),
        pseudo: pseudo.map(str::to_string),
        ..Default::default()
    })
    .expect("cell must build")
}

fn silicon() -> Cell {
    let h = 5.1311;
    let q = 2.55555;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("Si".into(), [0.0, 0.0, 0.0]), ("Si".into(), [q, q, q])],
        Some("gth-pade"),
    )
}

fn diamond() -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])],
        Some("gth-pade"),
    )
}

fn he_all_electron() -> Cell {
    let h = 2.834589;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        ..Default::default()
    })
    .expect("He cell must build")
}

fn cell_by_name(name: &str) -> Cell {
    match name {
        "si" | "silicon" => silicon(),
        "diamond" => diamond(),
        "he" => he_all_electron(),
        other => panic!("unknown --cell {other} (expected si|diamond|he)"),
    }
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

fn time_ms<T>(mut f: impl FnMut() -> T) -> (T, f64) {
    let t0 = Instant::now();
    let out = f();
    (out, t0.elapsed().as_secs_f64() * 1e3)
}

#[derive(Serialize, Default)]
struct JkReport {
    cell: String,
    nk: [usize; 3],
    mesh: [usize; 3],
    xc: String,
    hybrid: bool,
    nao: usize,
    nkpts: usize,
    ngrids: usize,
    warm_get_j_kpts_ms: f64,
    warm_get_k_kpts_ms: f64,
    /// `KNumInt::nr_rks` with the AO cache already populated — the pure-functional
    /// hot path (W-06/W-07 target).
    warm_nr_rks_ms: f64,
    /// The same call on a COLD AO cache: the one-off AO collocation.
    cold_nr_rks_ms: f64,
    get_ovlp_ms: f64,
    get_hcore_ms: f64,
    full_kernel_ms: f64,
    e_tot: f64,
    converged: bool,
}

#[derive(Serialize, Default)]
struct TransformReport {
    mesh: [usize; 3],
    n_batch: usize,
    n_transforms: usize,
    total_ms: f64,
    ns_per_grid_point: f64,
}

fn run_jk(args: &[String]) {
    let cell_name = arg_value(args, "--cell").unwrap_or_else(|| "si".into());
    let nk = parse_triple(&arg_value(args, "--nk").unwrap_or_else(|| "2,2,2".into()));
    let mesh = parse_triple(&arg_value(args, "--mesh").unwrap_or_else(|| "31,31,31".into()));
    let xc = arg_value(args, "--xc").unwrap_or_else(|| "pbe".into());
    let json_path = arg_value(args, "--json");
    let compare_path = arg_value(args, "--compare");

    let cell = cell_by_name(&cell_name);
    let kpts = make_kpts_default(&cell, nk).expect("k-mesh");
    let is_hybrid = hybrid(&xc).unwrap_or(false);

    // Warm pass: builds the AO table, the coulG/expmikr cache (W-01) and the
    // FFT plan cache, and gives us a converged density matrix to drive the
    // direct get_j_kpts/get_k_kpts timing below.
    let df_for_scf = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("FFTDF");
    let krks = Krks::from_df(Box::new(df_for_scf), &xc).expect("KRKS");
    let cfg = KScfConfig {
        conv_tol: 1e-9,
        max_cycle: 40,
        ..KScfConfig::default()
    };
    let (result, full_kernel_ms) = time_ms(|| krks.kernel(&cfg).expect("KRKS kernel"));

    // --- the one-off (per-SCF, not per-iteration) stages ---
    let (_, get_ovlp_ms) = time_ms(|| get_ovlp(&cell, &kpts).expect("get_ovlp"));
    // `get_hcore` is assembled by the density-fitting object (its V_loc,1 / V_nuc
    // term needs the FFT box), so it is timed on a fresh `Fftdf` — the one-off an
    // SCF pays before its first iteration.
    let df_hcore = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("FFTDF");
    let (_, get_hcore_ms) = time_ms(|| get_hcore(&df_hcore, &kpts).expect("get_hcore"));

    // --- nr_rks, cold and warm ---
    // `krks` still holds the KNumInt whose AO cache the SCF above populated, so
    // the warm number is what an SCF iteration actually pays; `ni.reset()` gives
    // the cold one (the AO collocation itself).
    krks.ni.reset();
    let (_, cold_nr_rks_ms) = time_ms(|| {
        krks.ni
            .nr_rks(&cell, &krks.grids, &xc, &result.dm, 1, None)
            .expect("cold nr_rks")
    });
    let (_, warm_nr_rks_ms) = time_ms(|| {
        krks.ni
            .nr_rks(&cell, &krks.grids, &xc, &result.dm, 1, None)
            .expect("warm nr_rks")
    });

    let df = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("FFTDF");
    // Prime the AO / coulG-expmikr caches before timing (RULE O: measure warm).
    let _ = get_j_kpts(&df, &result.dm, 1, &kpts, None, None).expect("warm get_j_kpts");
    if is_hybrid {
        let _ = get_k_kpts(&df, &result.dm, 1, &kpts, None, Some(ExxDiv::Ewald), None)
            .expect("warm get_k_kpts");
    }

    let (_, warm_j_ms) =
        time_ms(|| get_j_kpts(&df, &result.dm, 1, &kpts, None, None).expect("get_j_kpts"));
    let warm_k_ms = if is_hybrid {
        let (_, ms) = time_ms(|| {
            get_k_kpts(&df, &result.dm, 1, &kpts, None, Some(ExxDiv::Ewald), None)
                .expect("get_k_kpts")
        });
        ms
    } else {
        0.0
    };

    let report = JkReport {
        cell: cell_name,
        nk,
        mesh,
        xc: xc.clone(),
        hybrid: is_hybrid,
        nao: cell.mol.nao_nr,
        nkpts: kpts.len(),
        ngrids: df.ngrids(),
        warm_get_j_kpts_ms: warm_j_ms,
        warm_get_k_kpts_ms: warm_k_ms,
        warm_nr_rks_ms,
        cold_nr_rks_ms,
        get_ovlp_ms,
        get_hcore_ms,
        full_kernel_ms,
        e_tot: result.e_tot,
        converged: result.converged,
    };

    println!(
        "cell={} nk={:?} mesh={:?} xc={} hybrid={}\n  nao={} nkpts={} ngrids={}\n  \
         get_j_kpts (warm) = {:.3} ms\n  get_k_kpts (warm) = {:.3} ms\n  \
         nr_rks     (warm) = {:.3} ms   (cold = {:.3} ms)\n  \
         get_ovlp          = {:.3} ms\n  get_hcore         = {:.3} ms\n  \
         full kernel()     = {:.3} ms  (e_tot={:.10}, converged={})",
        report.cell,
        report.nk,
        report.mesh,
        report.xc,
        report.hybrid,
        report.nao,
        report.nkpts,
        report.ngrids,
        report.warm_get_j_kpts_ms,
        report.warm_get_k_kpts_ms,
        report.warm_nr_rks_ms,
        report.cold_nr_rks_ms,
        report.get_ovlp_ms,
        report.get_hcore_ms,
        report.full_kernel_ms,
        report.e_tot,
        report.converged,
    );

    if let Some(path) = &json_path {
        let s = serde_json::to_string_pretty(&report).expect("serialize report");
        std::fs::write(path, s).expect("write json");
        println!("wrote {path}");
    }

    if let Some(path) = &compare_path {
        let baseline: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}")),
        )
        .expect("parse baseline json");
        let base_j = baseline["warm_get_j_kpts_ms"].as_f64().unwrap_or(f64::NAN);
        let base_k = baseline["warm_get_k_kpts_ms"].as_f64().unwrap_or(f64::NAN);
        let base_nr = baseline["warm_nr_rks_ms"].as_f64().unwrap_or(f64::NAN);
        let base_full = baseline["full_kernel_ms"].as_f64().unwrap_or(f64::NAN);
        let base_e = baseline["e_tot"].as_f64().unwrap_or(f64::NAN);
        println!("compare vs {path}:");
        for (label, base, now) in [
            ("get_j_kpts", base_j, report.warm_get_j_kpts_ms),
            ("get_k_kpts", base_k, report.warm_get_k_kpts_ms),
            ("nr_rks    ", base_nr, report.warm_nr_rks_ms),
            ("kernel()  ", base_full, report.full_kernel_ms),
        ] {
            println!(
                "  {label}: {base:.3} ms -> {now:.3} ms ({:+.1}%)",
                100.0 * (now - base) / base
            );
        }
        // The energy is the accuracy signal: a speed change that moves `e_tot`
        // by more than the gate tolerance is a regression, not an optimisation.
        println!(
            "  e_tot     : {base_e:.12} -> {:.12}  (delta {:.3e})",
            report.e_tot,
            report.e_tot - base_e
        );
    }
}

/// The isolated `2 * Nk^2 * nao^2`-shaped transform batch of
/// KRKS-OPTIMISATION-PLAN.md §2.1 — the workload that attributed 93% of
/// `get_k_kpts`'s cost to the 3-D transform. Sweeps the meshes named in the
/// plan's own factorisation-cliff table (§2.1, §2.3) so a regression in the
/// mixed-radix/Rader plan selection (W-02) shows up here in seconds, not
/// minutes.
fn run_transform(args: &[String]) {
    use pyscf_algebra::CTensor;
    use pyscf_pbc_tools::fft::fft_stockham;

    let meshes_arg = arg_value(args, "--meshes").unwrap_or_else(|| "16,21,25,27,31,32".into());
    let n_batch: usize = arg_value(args, "--nbatch")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let json_path = arg_value(args, "--json");

    let mut reports = Vec::new();
    for tok in meshes_arg.split(',') {
        let n: usize = tok.trim().parse().expect("mesh axis must be an integer");
        let mesh = [n, n, n];
        let ngrids = n * n * n;
        let len = n_batch * ngrids;
        let mut re = vec![0.0_f64; len];
        let mut im = vec![0.0_f64; len];
        let mut s = 0x9E3779B97F4A7C15u64 ^ (n as u64);
        for v in re.iter_mut().chain(im.iter_mut()) {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *v = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        }
        let x = CTensor::from_planes(re, im);

        // Warm: pay for plan construction (the twiddle tables) once.
        let _ = fft_stockham(&x, mesh, false).expect("warm fft");

        let (_, ms) = time_ms(|| fft_stockham(&x, mesh, false).expect("fft"));
        let ns_per_point = ms * 1e6 / len as f64;
        println!(
            "mesh=[{n},{n},{n}] ngrids={ngrids} n_batch={n_batch}: {ms:.2} ms total, \
             {ns_per_point:.2} ns/grid-point"
        );
        reports.push(TransformReport {
            mesh,
            n_batch,
            n_transforms: n_batch,
            total_ms: ms,
            ns_per_grid_point: ns_per_point,
        });
    }

    if let Some(path) = &json_path {
        let s = serde_json::to_string_pretty(&reports).expect("serialize reports");
        std::fs::write(path, s).expect("write json");
        println!("wrote {path}");
    }
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_triple(s: &str) -> [usize; 3] {
    let parts: Vec<usize> = s.split(',').map(|t| t.trim().parse().expect("integer")).collect();
    assert_eq!(parts.len(), 3, "expected a,b,c, got {s}");
    [parts[0], parts[1], parts[2]]
}

fn main() {
    let all: Vec<String> = std::env::args().collect();
    let sub = all.get(1).cloned().unwrap_or_else(|| "transform".into());
    let rest = &all[all.len().min(2)..];
    match sub.as_str() {
        "transform" => run_transform(rest),
        "jk" => run_jk(rest),
        other => {
            eprintln!("unknown subcommand {other:?} (expected transform|jk)");
            std::process::exit(1);
        }
    }
}

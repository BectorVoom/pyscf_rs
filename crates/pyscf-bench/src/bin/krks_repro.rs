//! Determinism probe for `KRKS(Si, 2x2x2, PBE0)` — the case whose converged
//! energy was observed to move by 2 ulp between builds
//! (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md` §9 E-10).
//!
//! The accuracy gate is the wrong instrument for chasing that: it spends most
//! of its wall time shelling out to upstream PySCF, it runs seven SCFs
//! concurrently, and it prints one number per case. This runs the SAME setup as
//! `crates/pyscf-pbc-dft/tests/gate.rs::krks_si_222_pbe0_matches_upstream`
//! (`tight()`: `conv_tol = 1e-12`, `conv_tol_grad = 1e-8`, `max_cycle = 60`)
//! with no oracle, and reports the raw bits of every energy component plus the
//! CYCLE COUNT — which is the first thing to check, because a difference of one
//! SCF iteration explains a last-digit move without any reduction being at
//! fault.
//!
//! ```bash
//! # Same process, three times: intra-process determinism.
//! cargo run -p pyscf-bench --release --bin krks_repro -- --repeat 3
//!
//! # Three at once on separate threads: does concurrency change the answer?
//! # (This is what the gate does — seven tests share one rayon pool.)
//! cargo run -p pyscf-bench --release --bin krks_repro -- --concurrent 3
//!
//! # Thread-count dependence.
//! RAYON_NUM_THREADS=1 cargo run -p pyscf-bench --release --bin krks_repro
//! ```

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_scf::KScfConfig;

/// `tests/gate.rs::MESH_GATE`.
const MESH_GATE: [usize; 3] = [31, 31, 31];

/// `tests/gate.rs::silicon()` — geometry in BOHR, as that file insists.
fn silicon() -> Cell {
    let h = 5.1311;
    let q = 2.55555;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("Si".into(), [0.0, 0.0, 0.0]),
                ("Si".into(), [q, q, q]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("cell must build")
}

/// `tests/gate.rs::tight()`.
fn tight() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-8),
        max_cycle: 60,
        ..KScfConfig::default()
    }
}

struct Run {
    e_tot: f64,
    e_elec: f64,
    e_coul: f64,
    e_nuc: f64,
    cycles: u32,
    converged: bool,
    /// Raw bits of the first `mo_energy` block — the DM's fingerprint, so a
    /// difference can be localised to the orbitals rather than only the energy.
    mo0: Vec<u64>,
}

fn run_one(nk: [usize; 3], mesh: [usize; 3], xc: &str) -> Run {
    let cell = silicon();
    let kpts = make_kpts_default(&cell, nk).expect("k-mesh");
    let df = Fftdf::with_mesh(cell, &kpts, mesh).expect("FFTDF");
    let mf = Krks::from_df(Box::new(df), xc).expect("KRKS");
    let r = mf.kernel(&tight()).expect("KRKS kernel");
    Run {
        e_tot: r.e_tot,
        e_elec: r.e_elec,
        e_coul: r.e_coul,
        e_nuc: r.e_nuc,
        cycles: r.cycles,
        converged: r.converged,
        mo0: r.mo_energy[0].iter().map(|v| v.to_bits()).collect(),
    }
}

fn report(label: &str, r: &Run) {
    println!(
        "{label}: cycles={} converged={} \n    e_tot  {:.17e}  bits {:#018x}\n    \
         e_elec {:.17e}  bits {:#018x}\n    e_coul {:.17e}  bits {:#018x}\n    \
         e_nuc  {:.17e}  bits {:#018x}\n    mo_energy[0] bits {:?}",
        r.cycles,
        r.converged,
        r.e_tot,
        r.e_tot.to_bits(),
        r.e_elec,
        r.e_elec.to_bits(),
        r.e_coul,
        r.e_coul.to_bits(),
        r.e_nuc,
        r.e_nuc.to_bits(),
        r.mo0,
    );
}

/// Revisions of the UNPINNED sibling path dependencies.
///
/// `pyscf-dft` has `default = ["libxc"]`, so
/// `pyscf-pbc-dft -> pyscf-dft -> libxc_rs -> libxc-reval` is in the DEFAULT
/// dependency graph: the XC evaluator behind every gate residual lives in a
/// sibling working tree that this repository does not version. `cintx`
/// (integrals) and `xcfun_rs` are the same. A gate number can therefore move
/// with no commit here at all — which is exactly what happened on
/// 2026-08-31/09-01 — so an energy printed without saying which sibling trees
/// produced it is not a reproducible measurement.
fn provenance() {
    for repo in ["../libxc_rs", "../cintx", "../xcfun_rs"] {
        let rev = std::process::Command::new("git")
            .args([
                "-C",
                repo,
                "log",
                "-1",
                "--format=%h %ad %s",
                "--date=short",
            ])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map_or_else(
                || "<not a git tree>".to_string(),
                |o| String::from_utf8_lossy(&o.stdout).trim().to_string(),
            );
        let dirty = std::process::Command::new("git")
            .args(["-C", repo, "status", "--porcelain"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map_or(0, |o| String::from_utf8_lossy(&o.stdout).lines().count());
        println!("  dep {repo:<12} {rev}  [{dirty} uncommitted file(s)]");
    }
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let all: Vec<String> = std::env::args().collect();
    let xc = arg(&all, "--xc").unwrap_or_else(|| "pbe0".into());
    let nk = arg(&all, "--nk").map_or([2, 2, 2], |s| {
        let v: Vec<usize> = s
            .split(',')
            .map(|t| t.trim().parse().expect("int"))
            .collect();
        [v[0], v[1], v[2]]
    });
    let mesh = arg(&all, "--mesh").map_or(MESH_GATE, |s| {
        let v: Vec<usize> = s
            .split(',')
            .map(|t| t.trim().parse().expect("int"))
            .collect();
        [v[0], v[1], v[2]]
    });
    let repeat: usize = arg(&all, "--repeat")
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let concurrent: usize = arg(&all, "--concurrent")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    println!(
        "xc={xc} nk={nk:?} mesh={mesh:?} rayon_threads={} repeat={repeat} concurrent={concurrent}",
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get)
    );
    provenance();

    if concurrent > 0 {
        // Mimic the gate: several independent SCFs sharing one process and one
        // rayon pool. If the answer moves only here, the cause is contention,
        // not the arithmetic.
        let runs: Vec<Run> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..concurrent)
                .map(|_| s.spawn(|| run_one(nk, mesh, &xc)))
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().expect("thread"))
                .collect()
        });
        for (i, r) in runs.iter().enumerate() {
            report(&format!("concurrent[{i}]"), r);
        }
        let first = runs[0].e_tot.to_bits();
        println!(
            "concurrent agreement: {}",
            if runs.iter().all(|r| r.e_tot.to_bits() == first) {
                "ALL IDENTICAL"
            } else {
                "*** DIFFER ***"
            }
        );
        return;
    }

    let mut seen: Vec<u64> = Vec::new();
    for i in 0..repeat {
        let r = run_one(nk, mesh, &xc);
        report(&format!("run[{i}]"), &r);
        seen.push(r.e_tot.to_bits());
    }
    println!(
        "repeat agreement: {}",
        if seen.windows(2).all(|w| w[0] == w[1]) {
            "ALL IDENTICAL"
        } else {
            "*** DIFFER ***"
        }
    );
}

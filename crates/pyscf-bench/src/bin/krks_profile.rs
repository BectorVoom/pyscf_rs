//! W-00 (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`) — the committed,
//! re-runnable KRKS SCF profiling harness. Replaces the throwaway pair of
//! examples §2.1 of that plan was measured with (deleted after that run).
//!
//! ```bash
//! # The KUKS half (U-01): adds the nset=2 timings and nr_uks beside nr_rks.
//! cargo run -p pyscf-bench --release --bin krks_profile -- \
//!     jk --driver kuks --cell si --nk 2,2,2 --mesh 31,31,31 --xc pbe0
//!
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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_df::{Fftdf, Gdf, PeriodicDf, get_hcore, get_j_kpts, get_k_kpts, get_k_kpts_opts};
use pyscf_pbc_dft::krks::{Krks, hybrid};
use pyscf_pbc_dft::kuks::Kuks;
use pyscf_pbc_gto::hcore::get_ovlp;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_scf::KScfConfig;
use pyscf_pbc_tools::coulg::ExxDiv;
use serde::Serialize;
use tracing::{Subscriber, field::Visit, span::Id};
use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

// ---------------------------------------------------------------------------
// Cells — same geometries as `pyscf-pbc-dft/tests/common/mod.rs` (not a
// public API of that crate, so reproduced here rather than depended on).
// ---------------------------------------------------------------------------

fn bohr_cell(
    a: [[f64; 3]; 3],
    atoms: Vec<(String, [f64; 3])>,
    pseudo: Option<&str>,
    basis: &str,
) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(a),
        pseudo: pseudo.map(str::to_string),
        ..Default::default()
    })
    .expect("cell must build")
}

fn silicon(basis: &str) -> Cell {
    let h = 5.1311;
    let q = 2.55555;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("Si".into(), [0.0, 0.0, 0.0]), ("Si".into(), [q, q, q])],
        Some("gth-pade"),
        basis,
    )
}

/// A Si 2x2x1 supercell — W-09's "large cell" (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`),
/// the baseline its DEFER UNTIL clause asks for. Four times the atoms of the
/// reference cell, so `nao` and the AO table grow with it while the k-mesh can
/// shrink to keep the run affordable.
fn silicon_supercell(basis: &str) -> Cell {
    let h = 5.1311;
    let q = 2.55555;
    // Two primitive cells stacked along a1 + a2.
    let a1 = [0.0, h, h];
    let a2 = [h, 0.0, h];
    let a3 = [h, h, 0.0];
    let double = |v: [f64; 3]| [2.0 * v[0], 2.0 * v[1], 2.0 * v[2]];
    let mut atoms = Vec::new();
    for (i, j) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
        let shift = [
            i * a1[0] + j * a2[0],
            i * a1[1] + j * a2[1],
            i * a1[2] + j * a2[2],
        ];
        atoms.push(("Si".into(), shift));
        atoms.push(("Si".into(), [shift[0] + q, shift[1] + q, shift[2] + q]));
    }
    bohr_cell([double(a1), double(a2), a3], atoms, Some("gth-pade"), basis)
}

fn diamond(basis: &str) -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    bohr_cell(
        [[0.0, h, h], [h, 0.0, h], [h, h, 0.0]],
        vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])],
        Some("gth-pade"),
        basis,
    )
}

fn he_all_electron(basis: &str) -> Cell {
    let h = 2.834589;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        ..Default::default()
    })
    .expect("He cell must build")
}

fn cell_by_name(name: &str, basis: &str) -> Cell {
    match name {
        "si" | "silicon" => silicon(basis),
        "si-supercell" => silicon_supercell(basis),
        "diamond" => diamond(basis),
        "he" => he_all_electron(if basis == "gth-szv" { "sto-3g" } else { basis }),
        "li" => Cell::build(CellBuildArgs {
            mole: MoleBuildArgs {
                atom: AtomInput::Tuples(vec![("Li".into(), [0.0, 0.0, 0.0])]),
                basis: BasisInput::Name("sto-3g".into()),
                unit: Unit::Bohr,
                spin: 1,
                ..Default::default()
            },
            a: ALattice::Matrix([[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 6.0]]),
            ..Default::default()
        })
        .expect("Li cell must build"),
        other => panic!("unknown --cell {other} (expected si|si-supercell|diamond|he|li)"),
    }
}

#[derive(Serialize)]
struct MultigridReport {
    driver: String,
    numint: String,
    cell: String,
    mesh: [usize; 3],
    kernel_ms: f64,
    warm_get_veff_ms: f64,
    build_tasks_cache_miss_ms: f64,
    build_tasks_cache_hit_ms: f64,
    /// v1 only: level-value collocation inside the WHOLE `kernel()`, and the
    /// number of times it ran (one per cycle before session 3's cache).
    kernel_collocate_ms: f64,
    kernel_collocate_calls: u64,
    /// The warm `get_veff`'s collocation and shared Coulomb/XC middle.
    warm_collocate_ms: f64,
    warm_collocate_calls: u64,
    warm_xc_parts_ms: f64,
    warm_levels: Vec<MultigridLevelReport>,
    e_tot: f64,
    converged: bool,
    max_dm_spin_delta: Option<f64>,
    peak_rss_mib: f64,
    load_average_at_start: f64,
}

#[derive(Clone, Default, Serialize)]
struct MultigridLevelReport {
    level: usize,
    forward_ms: f64,
    reverse_ms: f64,
    forward_fft_ms: f64,
    reverse_fft_ms: f64,
    forward_launches: u64,
    reverse_launches: u64,
    transfer_bytes_before: u64,
    transfer_bytes_after: u64,
}

fn run_multigrid(args: &[String]) {
    use pyscf_pbc_dft::numint::KsNumInt;

    let driver = arg_value(args, "--driver").unwrap_or_else(|| "krks".into());
    let numint = arg_value(args, "--numint").unwrap_or_else(|| "v1".into());
    assert!(matches!(driver.as_str(), "krks" | "kuks"));
    assert!(matches!(numint.as_str(), "grid" | "v1" | "v2"));
    let cell_name = arg_value(args, "--cell").unwrap_or_else(|| {
        if driver == "kuks" {
            "li".into()
        } else {
            "si".into()
        }
    });
    let mesh = parse_triple(&arg_value(args, "--mesh").unwrap_or_else(|| "11,11,11".into()));
    let load0 = load_average();
    let json_path = arg_value(args, "--json");
    if let Some(path) = json_path.as_deref() {
        guard_idle_baseline_write(path, load0);
    }
    let mut cell = cell_by_name(&cell_name, "gth-szv");
    cell.mesh = mesh;
    let ni = || match numint.as_str() {
        "grid" => KsNumInt::grid(&[[0.0; 3]]),
        "v1" => KsNumInt::multigrid(),
        "v2" => KsNumInt::multigrid2(),
        _ => unreachable!(),
    };
    let cfg = KScfConfig {
        conv_tol: 1e-9,
        max_cycle: 40,
        ..Default::default()
    };
    let kernel_stages = MgStageLayer::default();
    let warm_stages = MgStageLayer::default();
    let (result, kernel_ms, warm_ms, spin_delta) = if driver == "kuks" {
        let mut mf = Kuks::new(cell, &[[0.0; 3]], "lda,vwn").expect("KUKS");
        mf.ni = ni();
        let (result, kernel_ms) = tracing::subscriber::with_default(
            tracing_subscriber::registry().with(kernel_stages.clone()),
            || time_ms(|| mf.kernel(&cfg).expect("KUKS kernel")),
        );
        let (_, warm_ms) = tracing::subscriber::with_default(
            tracing_subscriber::registry().with(warm_stages.clone()),
            || time_ms(|| mf.get_veff_tagged(&result.dm, None).expect("get_veff")),
        );
        let delta = result.dm[0]
            .iter()
            .zip(&result.dm[1])
            .flat_map(|(dma, dmb)| dma.re.iter().zip(&dmb.re))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        (result, kernel_ms, warm_ms, Some(delta))
    } else {
        let mut mf = Krks::new(cell, &[[0.0; 3]], "lda,vwn").expect("KRKS");
        mf.ni = ni();
        let (result, kernel_ms) = tracing::subscriber::with_default(
            tracing_subscriber::registry().with(kernel_stages.clone()),
            || time_ms(|| mf.kernel(&cfg).expect("KRKS kernel")),
        );
        let (_, warm_ms) = tracing::subscriber::with_default(
            tracing_subscriber::registry().with(warm_stages.clone()),
            || time_ms(|| mf.get_veff_tagged(&result.dm, None).expect("get_veff")),
        );
        (result, kernel_ms, warm_ms, None)
    };
    let kernel_timing = kernel_stages.snapshot();
    let warm_timing = warm_stages.snapshot();
    let report = MultigridReport {
        driver,
        numint,
        cell: cell_name,
        mesh,
        kernel_ms,
        warm_get_veff_ms: warm_ms,
        build_tasks_cache_miss_ms: kernel_timing.build_miss_ms,
        build_tasks_cache_hit_ms: warm_timing.build_hit_ms,
        kernel_collocate_ms: kernel_timing.collocate_ms,
        kernel_collocate_calls: kernel_timing.collocate_calls,
        warm_collocate_ms: warm_timing.collocate_ms,
        warm_collocate_calls: warm_timing.collocate_calls,
        warm_xc_parts_ms: warm_timing.xc_parts_ms,
        warm_levels: warm_timing.levels,
        e_tot: result.e_tot,
        converged: result.converged,
        max_dm_spin_delta: spin_delta,
        peak_rss_mib: peak_rss_mib(),
        load_average_at_start: load0,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize")
    );
    if let Some(path) = json_path {
        std::fs::write(
            path,
            serde_json::to_string_pretty(&report).expect("serialize"),
        )
        .expect("write json");
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

const AO_STAGE_NAMES: [&str; 4] = [
    "pbc_eval_ao_shift_pack",
    "pbc_eval_ao_eval_gto",
    "pbc_eval_ao_scatter",
    "pbc_eval_ao_k08_accumulate",
];

#[derive(Clone, Default)]
struct AoStageLayer {
    inner: Arc<AoStageLayerInner>,
}

#[derive(Default)]
struct AoStageLayerInner {
    entered: Mutex<HashMap<Id, Instant>>,
    elapsed: Mutex<[Duration; 4]>,
    /// `(launched images, kept grid points summed over them)` — read off the
    /// `points` field of every `pbc_eval_ao_eval_gto` span. The screen's
    /// actual yield, which `nimgs` (the image LIST length) cannot show.
    launches: Mutex<(u64, u64)>,
}

#[derive(Default)]
struct AoPointsVisitor {
    points: Option<u64>,
}

impl Visit for AoPointsVisitor {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        if field.name() == "points" {
            self.points = Some(value);
        }
    }

    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
}

impl AoStageLayer {
    fn milliseconds(&self) -> [f64; 4] {
        let elapsed = self.inner.elapsed.lock().expect("AO stage elapsed mutex");
        elapsed.map(|d| d.as_secs_f64() * 1e3)
    }

    fn launches(&self) -> (u64, u64) {
        *self.inner.launches.lock().expect("AO launches mutex")
    }
}

impl<S> Layer<S> for AoStageLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, _id: &Id, _ctx: Context<'_, S>) {
        if attrs.metadata().name() != AO_STAGE_NAMES[1] {
            return;
        }
        let mut visitor = AoPointsVisitor::default();
        attrs.record(&mut visitor);
        let mut launches = self.inner.launches.lock().expect("AO launches mutex");
        launches.0 += 1;
        launches.1 += visitor.points.unwrap_or(0);
    }

    fn on_enter(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        if AO_STAGE_NAMES.contains(&span.metadata().name()) {
            self.inner
                .entered
                .lock()
                .expect("AO stage entered mutex")
                .insert(id.clone(), Instant::now());
        }
    }

    fn on_exit(&self, id: &Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(id) else { return };
        let Some(stage) = AO_STAGE_NAMES
            .iter()
            .position(|name| *name == span.metadata().name())
        else {
            return;
        };
        let Some(started) = self
            .inner
            .entered
            .lock()
            .expect("AO stage entered mutex")
            .remove(id)
        else {
            return;
        };
        self.inner.elapsed.lock().expect("AO stage elapsed mutex")[stage] += started.elapsed();
    }
}

/// Counts every cold `eval_ao_kpts` (the `pbc_eval_ao_kpts` span) inside a
/// timed region: how many AO tables a driver builds, at how many k-points
/// each, and their summed wall time. S-07's instrument.
#[derive(Clone, Default)]
struct AoCallLayer {
    inner: Arc<AoCallLayerInner>,
}

#[derive(Default)]
struct AoCallLayerInner {
    entered: Mutex<HashMap<Id, (Instant, u64)>>,
    calls: Mutex<Vec<(u64, f64)>>,
}

#[derive(Default)]
struct AoCallVisitor {
    nkpts: Option<u64>,
}

impl Visit for AoCallVisitor {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        if field.name() == "nkpts" {
            self.nkpts = Some(value);
        }
    }

    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
}

impl AoCallLayer {
    /// `(calls, total ms, k-point count of every call)`.
    fn summary(&self) -> (u64, f64, Vec<u64>) {
        let calls = self.inner.calls.lock().expect("AO call mutex");
        (
            calls.len() as u64,
            calls.iter().map(|c| c.1).sum(),
            calls.iter().map(|c| c.0).collect(),
        )
    }
}

impl<S> Layer<S> for AoCallLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &Id, _ctx: Context<'_, S>) {
        if attrs.metadata().name() != "pbc_eval_ao_kpts" {
            return;
        }
        let mut visitor = AoCallVisitor::default();
        attrs.record(&mut visitor);
        self.inner
            .entered
            .lock()
            .expect("AO call mutex")
            .insert(id.clone(), (Instant::now(), visitor.nkpts.unwrap_or(0)));
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        let Some((started, nkpts)) = self.inner.entered.lock().expect("AO call mutex").remove(&id)
        else {
            return;
        };
        self.inner
            .calls
            .lock()
            .expect("AO call mutex")
            .push((nkpts, started.elapsed().as_secs_f64() * 1e3));
    }
}

const MG_STAGE_NAMES: [&str; 7] = [
    "pbc_mg_build_tasks_hit",
    "pbc_mg_build_tasks_miss",
    "pbc_mg_forward_level",
    "pbc_mg_reverse_level",
    "pbc_mg_fft",
    // Session 3: the two v1 stages that sat OUTSIDE the level spans — the
    // per-call re-collocation and the shared Coulomb/XC middle.
    "pbc_mg_collocate",
    "pbc_mg_xc_parts",
];

#[derive(Clone, Default)]
struct MgStageLayer {
    inner: Arc<MgStageLayerInner>,
}

#[derive(Default)]
struct MgStageLayerInner {
    entered: Mutex<HashMap<Id, Instant>>,
    metadata: Mutex<HashMap<Id, MgSpanMetadata>>,
    timing: Mutex<MgTiming>,
}

#[derive(Clone, Default)]
struct MgSpanMetadata {
    name: &'static str,
    level: usize,
    launches: u64,
    transfer_bytes_before: u64,
    transfer_bytes_after: u64,
    direction: String,
}

#[derive(Default)]
struct MgFieldVisitor {
    level: Option<u64>,
    launches: Option<u64>,
    transfer_bytes_before: Option<u64>,
    transfer_bytes_after: Option<u64>,
    direction: Option<String>,
}

impl Visit for MgFieldVisitor {
    fn record_u64(&mut self, field: &tracing::field::Field, value: u64) {
        match field.name() {
            "level" => self.level = Some(value),
            "launches" => self.launches = Some(value),
            "transfer_bytes_before" => self.transfer_bytes_before = Some(value),
            "transfer_bytes_after" => self.transfer_bytes_after = Some(value),
            _ => {}
        }
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "direction" {
            self.direction = Some(value.to_owned());
        }
    }

    fn record_debug(&mut self, _field: &tracing::field::Field, _value: &dyn std::fmt::Debug) {}
}

#[derive(Clone, Default)]
struct MgTiming {
    build_hit_ms: f64,
    build_miss_ms: f64,
    collocate_ms: f64,
    collocate_calls: u64,
    xc_parts_ms: f64,
    levels: Vec<MultigridLevelReport>,
}

impl MgStageLayer {
    fn snapshot(&self) -> MgTiming {
        self.inner.timing.lock().expect("MG timing mutex").clone()
    }
}

impl<S> Layer<S> for MgStageLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &Id, _ctx: Context<'_, S>) {
        let name = attrs.metadata().name();
        if !MG_STAGE_NAMES.contains(&name) {
            return;
        }
        let mut visitor = MgFieldVisitor::default();
        attrs.record(&mut visitor);
        self.inner
            .metadata
            .lock()
            .expect("MG metadata mutex")
            .insert(
                id.clone(),
                MgSpanMetadata {
                    name,
                    level: visitor.level.unwrap_or(0) as usize,
                    launches: visitor.launches.unwrap_or(0),
                    transfer_bytes_before: visitor.transfer_bytes_before.unwrap_or(0),
                    transfer_bytes_after: visitor.transfer_bytes_after.unwrap_or(0),
                    direction: visitor.direction.unwrap_or_default(),
                },
            );
    }

    fn on_enter(&self, id: &Id, _ctx: Context<'_, S>) {
        if self
            .inner
            .metadata
            .lock()
            .expect("MG metadata mutex")
            .contains_key(id)
        {
            self.inner
                .entered
                .lock()
                .expect("MG entered mutex")
                .insert(id.clone(), Instant::now());
        }
    }

    fn on_exit(&self, id: &Id, _ctx: Context<'_, S>) {
        let Some(started) = self
            .inner
            .entered
            .lock()
            .expect("MG entered mutex")
            .remove(id)
        else {
            return;
        };
        let Some(meta) = self
            .inner
            .metadata
            .lock()
            .expect("MG metadata mutex")
            .get(id)
            .cloned()
        else {
            return;
        };
        let elapsed_ms = started.elapsed().as_secs_f64() * 1e3;
        let mut timing = self.inner.timing.lock().expect("MG timing mutex");
        match meta.name {
            "pbc_mg_build_tasks_hit" => timing.build_hit_ms += elapsed_ms,
            "pbc_mg_build_tasks_miss" => timing.build_miss_ms += elapsed_ms,
            "pbc_mg_collocate" => {
                timing.collocate_ms += elapsed_ms;
                timing.collocate_calls += 1;
            }
            "pbc_mg_xc_parts" => timing.xc_parts_ms += elapsed_ms,
            _ => {
                if timing.levels.len() <= meta.level {
                    timing
                        .levels
                        .resize_with(meta.level + 1, MultigridLevelReport::default);
                }
                let level = &mut timing.levels[meta.level];
                level.level = meta.level;
                match meta.name {
                    "pbc_mg_forward_level" => {
                        level.forward_ms += elapsed_ms;
                        level.forward_launches += meta.launches;
                        level.transfer_bytes_before = level
                            .transfer_bytes_before
                            .saturating_add(meta.transfer_bytes_before);
                        level.transfer_bytes_after = level
                            .transfer_bytes_after
                            .saturating_add(meta.transfer_bytes_after);
                    }
                    "pbc_mg_reverse_level" => {
                        level.reverse_ms += elapsed_ms;
                        level.reverse_launches += meta.launches;
                        level.transfer_bytes_before = level
                            .transfer_bytes_before
                            .saturating_add(meta.transfer_bytes_before);
                        level.transfer_bytes_after = level
                            .transfer_bytes_after
                            .saturating_add(meta.transfer_bytes_after);
                    }
                    "pbc_mg_fft" if meta.direction == "forward" => {
                        level.forward_fft_ms += elapsed_ms;
                    }
                    "pbc_mg_fft" if meta.direction == "reverse" => {
                        level.reverse_fft_ms += elapsed_ms;
                    }
                    _ => {}
                }
            }
        }
    }

    fn on_close(&self, id: Id, _ctx: Context<'_, S>) {
        self.inner
            .metadata
            .lock()
            .expect("MG metadata mutex")
            .remove(&id);
    }
}

#[derive(Serialize, Default)]
struct JkReport {
    cell: String,
    basis: String,
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
    // -----------------------------------------------------------------
    // U-01 (`.planning/pbc/KUKS-OPTIMISATION-PLAN.md`) — the `nset = 2` half.
    //
    // §2.1.2 could only BOUND the KUKS/KRKS multiplier at `1 < m < 2`, because
    // the shares it inherited from the KRKS plan's §2.1 predate W-02/W-02b and
    // W-06/W-07 and are therefore stale in the one direction that matters: the
    // transform got both algorithmically cheaper and parallel while the
    // contractions — the part KUKS doubles — did not. U-01 step 2 says the
    // cheapest useful measurement needs no new cell and no new physics: call
    // the SAME `get_j_kpts`/`get_k_kpts` on `[dm, dm]` and on `[dm]`, and the
    // ratio of the two IS the multiplier, on identical data.
    // -----------------------------------------------------------------
    /// Which driver's SCF produced `e_tot`/`full_kernel_ms`: `krks` or `kuks`.
    driver: String,
    /// `get_j_kpts` on a two-set density built from the converged one.
    warm_get_j_kpts_nset2_ms: f64,
    /// `get_k_kpts` on the same, `0.0` for a pure functional.
    warm_get_k_kpts_nset2_ms: f64,
    /// `nset = 2 / nset = 1` on identical data — the measured multiplier.
    nset2_over_nset1_j: f64,
    nset2_over_nset1_k: f64,
    /// `KNumInt::nr_uks` warm / cold, beside `nr_rks` on the same cell.
    warm_nr_uks_ms: f64,
    cold_nr_uks_ms: f64,
    /// `nr_uks / nr_rks`, warm.
    nr_uks_over_nr_rks: f64,
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
    // W-09's DEFER UNTIL clause asks for a LARGE-CELL baseline; `--basis` and
    // `--cell si-supercell` are how this harness produces one.
    let basis = arg_value(args, "--basis").unwrap_or_else(|| "gth-szv".into());
    let nk = parse_triple(&arg_value(args, "--nk").unwrap_or_else(|| "2,2,2".into()));
    let mesh = parse_triple(&arg_value(args, "--mesh").unwrap_or_else(|| "31,31,31".into()));
    let xc = arg_value(args, "--xc").unwrap_or_else(|| "pbe".into());
    // U-01 step 1: the harness was KRKS-only.
    let driver = arg_value(args, "--driver").unwrap_or_else(|| "krks".into());
    assert!(
        driver == "krks" || driver == "kuks",
        "--driver must be krks or kuks, got {driver:?}"
    );
    let json_path = arg_value(args, "--json");
    let compare_path = arg_value(args, "--compare");

    let cell = cell_by_name(&cell_name, &basis);
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
    // The driver under test supplies `e_tot` and `full_kernel_ms`; the KRKS
    // object is built either way because its `KNumInt` and grids are what the
    // per-stage timings below reuse, and reusing ONE grid/AO cache for both
    // `nr_rks` and `nr_uks` is what makes their ratio a measurement of the
    // spin doubling rather than of two different caches.
    let (result, full_kernel_ms) = if driver == "kuks" {
        let df_u = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("FFTDF");
        let kuks = Kuks::from_df(Box::new(df_u), &xc).expect("KUKS");
        let (r, ms) = time_ms(|| kuks.kernel(&cfg).expect("KUKS kernel"));
        // Warm the shared KRKS caches too, so the stage timings below are warm.
        let _ = krks.kernel(&cfg).expect("KRKS kernel");
        (r, ms)
    } else {
        time_ms(|| krks.kernel(&cfg).expect("KRKS kernel"))
    };
    // `nset = 1` k-stack, whatever the driver was: the stage timings are all
    // expressed against it so the `nset = 2` ratio is like-for-like.
    let dm1: Vec<Vec<pyscf_algebra::CTensor>> = vec![result.dm[0].clone()];
    // U-01 step 2: the SAME density in both channels. This measures the
    // `nset` doubling and nothing else — no second SCF, no second cell, and
    // no second physical state to explain a difference away with.
    let dm2: Vec<Vec<pyscf_algebra::CTensor>> = vec![result.dm[0].clone(), result.dm[0].clone()];

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
            .nr_rks(&cell, &krks.grids, &xc, &dm1, 1, &kpts, None)
            .expect("cold nr_rks")
    });
    let (_, warm_nr_rks_ms) = time_ms(|| {
        krks.ni
            .nr_rks(&cell, &krks.grids, &xc, &dm1, 1, &kpts, None)
            .expect("warm nr_rks")
    });

    // U-01 step 3: `nr_uks` beside `nr_rks`, same cell, same grid, same AO
    // cache, same density in both channels. §2.1.1 predicts "just under 2x" —
    // `eval_ao` is shared once per block, `eval_rho` and `accumulate_vxc` run
    // twice, and `eval_xc_eff_uks` is one call on a 2x-wider kernel.
    let uks_sets: [Vec<Vec<pyscf_algebra::CTensor>>; 2] = [dm1.clone(), dm1.clone()];
    krks.ni.reset();
    let (_, cold_nr_uks_ms) = time_ms(|| {
        krks.ni
            .nr_uks(&cell, &krks.grids, &xc, &uks_sets, 1, &kpts, None)
            .expect("cold nr_uks")
    });
    let (_, warm_nr_uks_ms) = time_ms(|| {
        krks.ni
            .nr_uks(&cell, &krks.grids, &xc, &uks_sets, 1, &kpts, None)
            .expect("warm nr_uks")
    });

    let df = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("FFTDF");
    // Prime the AO / coulG-expmikr caches before timing (RULE O: measure warm).
    let _ = get_j_kpts(&df, &dm1, 1, &kpts, None, None).expect("warm get_j_kpts");
    if is_hybrid {
        let _ = get_k_kpts(&df, &dm1, 1, &kpts, None, Some(ExxDiv::Ewald), None)
            .expect("warm get_k_kpts");
    }

    let (_, warm_j_ms) =
        time_ms(|| get_j_kpts(&df, &dm1, 1, &kpts, None, None).expect("get_j_kpts"));
    let _ = get_j_kpts(&df, &dm2, 1, &kpts, None, None).expect("warm get_j_kpts nset=2");
    let (_, warm_j_nset2_ms) =
        time_ms(|| get_j_kpts(&df, &dm2, 1, &kpts, None, None).expect("get_j_kpts nset=2"));
    // W-08: `--kk-symmetry` times the halved pair loop instead of the full one.
    // It is opt-in because it changes the last bits of `vk`; timing it here
    // beside the full loop is how its 1.x is attributed.
    let kk_symmetry = args.iter().any(|a| a == "--kk-symmetry");
    let (warm_k_ms, warm_k_nset2_ms) = if is_hybrid {
        let mut run = |dms: &Vec<Vec<pyscf_algebra::CTensor>>| {
            let _ = get_k_kpts_opts(
                &df,
                dms,
                1,
                &kpts,
                None,
                Some(ExxDiv::Ewald),
                None,
                kk_symmetry,
            )
            .expect("warm get_k_kpts");
            let (_, ms) = time_ms(|| {
                get_k_kpts_opts(
                    &df,
                    dms,
                    1,
                    &kpts,
                    None,
                    Some(ExxDiv::Ewald),
                    None,
                    kk_symmetry,
                )
                .expect("get_k_kpts")
            });
            ms
        };
        (run(&dm1), run(&dm2))
    } else {
        (0.0, 0.0)
    };

    let report = JkReport {
        cell: cell_name,
        basis: basis.clone(),
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
        driver: driver.clone(),
        warm_get_j_kpts_nset2_ms: warm_j_nset2_ms,
        warm_get_k_kpts_nset2_ms: warm_k_nset2_ms,
        nset2_over_nset1_j: warm_j_nset2_ms / warm_j_ms,
        nset2_over_nset1_k: if warm_k_ms > 0.0 {
            warm_k_nset2_ms / warm_k_ms
        } else {
            f64::NAN
        },
        warm_nr_uks_ms,
        cold_nr_uks_ms,
        nr_uks_over_nr_rks: warm_nr_uks_ms / warm_nr_rks_ms,
        get_ovlp_ms,
        get_hcore_ms,
        full_kernel_ms,
        e_tot: result.e_tot,
        converged: result.converged,
    };

    println!(
        "cell={} basis={} nk={:?} mesh={:?} xc={} hybrid={}\n  nao={} nkpts={} ngrids={}\n  \
         get_j_kpts (warm) = {:.3} ms\n  get_k_kpts (warm) = {:.3} ms\n  \
         nr_rks     (warm) = {:.3} ms   (cold = {:.3} ms)\n  \
         get_ovlp          = {:.3} ms\n  get_hcore         = {:.3} ms\n  \
         full kernel()     = {:.3} ms  (e_tot={:.10}, converged={})",
        report.cell,
        report.basis,
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
    // U-01: the `nset = 2` block, printed beside the `nset = 1` one so the
    // multiplier is read off directly rather than inferred from two runs.
    println!(
        "  --- U-01, nset = 2 on IDENTICAL data (driver={}) ---\n  \
         get_j_kpts nset=2 = {:.3} ms   ({:.3}x nset=1)\n  \
         get_k_kpts nset=2 = {:.3} ms   ({:.3}x nset=1)\n  \
         nr_uks     (warm) = {:.3} ms   (cold = {:.3} ms, {:.3}x nr_rks)",
        report.driver,
        report.warm_get_j_kpts_nset2_ms,
        report.nset2_over_nset1_j,
        report.warm_get_k_kpts_nset2_ms,
        report.nset2_over_nset1_k,
        report.warm_nr_uks_ms,
        report.cold_nr_uks_ms,
        report.nr_uks_over_nr_rks,
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
            (
                "nr_uks    ",
                baseline["warm_nr_uks_ms"].as_f64().unwrap_or(f64::NAN),
                report.warm_nr_uks_ms,
            ),
            (
                "get_j n=2 ",
                baseline["warm_get_j_kpts_nset2_ms"]
                    .as_f64()
                    .unwrap_or(f64::NAN),
                report.warm_get_j_kpts_nset2_ms,
            ),
            (
                "get_k n=2 ",
                baseline["warm_get_k_kpts_nset2_ms"]
                    .as_f64()
                    .unwrap_or(f64::NAN),
                report.warm_get_k_kpts_nset2_ms,
            ),
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

#[derive(Serialize, Default)]
struct ContractReport {
    shape: String,
    nao: usize,
    ngrids: usize,
    gflop: f64,
    host_ms: f64,
    device_zgemm_ms: f64,
    host_gflops: f64,
    device_gflops: f64,
    max_abs_diff: f64,
}

/// W-03 / W-06 decision benchmark (`.planning/pbc/KRKS-OPTIMISATION-PLAN.md`).
///
/// Both items say "route these contractions through `pyscf_algebra::zgemm_dense`",
/// and W-03 step 4 says the item MUST NOT land if the per-call upload/read-back
/// makes it slower than the host route. That is a MEASUREMENT, and this is the
/// measurement: the two shapes that actually dominate `get_k_kpts` and `nr_rks`,
/// run both ways on the same data, with the max absolute difference reported so a
/// speed comparison can never be quoted without the accuracy one.
///
/// * `nao x nao . nao x Ng` — `dm_times_conj_ao` (`fft_jk.rs`), `eval_rho_one`'s
///   `c0` stage (`numint.rs`). Reduction over `nao` (small).
/// * `nao x Ng . Ng x nao` — `accumulate_vk` (`fft_jk.rs`), `vxc_mat_one`'s outer
///   stage (`numint.rs`). Reduction over the GRID, which is the axis D-PBC-17
///   requires an ordered reduction on — so the host route here is `oracle_dot`,
///   not a naive `+=`.
fn run_contract(args: &[String]) {
    use pyscf_algebra::{CTensor, oracle_dot, select_backend, zgemm_dense};
    use rayon::prelude::*;

    let mesh = parse_triple(&arg_value(args, "--mesh").unwrap_or_else(|| "21,21,21".into()));
    let nao: usize = arg_value(args, "--nao")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let reps: usize = arg_value(args, "--reps")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let json_path = arg_value(args, "--json");

    let ngrids = mesh[0] * mesh[1] * mesh[2];
    let mut s = 0x9E37_79B9_7F4A_7C15u64;
    let mut rand = move || {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let ao = CTensor::from_planes(
        (0..nao * ngrids).map(|_| rand()).collect(),
        (0..nao * ngrids).map(|_| rand()).collect(),
    );
    let dm = CTensor::from_planes(
        (0..nao * nao).map(|_| rand()).collect(),
        (0..nao * nao).map(|_| rand()).collect(),
    );
    let client = select_backend().expect("backend").client;

    let mut reports = Vec::new();

    // ---- shape 1: (nao,nao) . (nao,Ng), reduction over nao -----------------
    {
        // 8 flops per complex FMA.
        let gflop = 8.0 * (nao * nao * ngrids) as f64 * 1e-9;
        let host = || {
            let mut re = vec![0.0_f64; nao * ngrids];
            let mut im = vec![0.0_f64; nao * ngrids];
            re.par_chunks_mut(ngrids)
                .zip(im.par_chunks_mut(ngrids))
                .enumerate()
                .for_each(|(j, (orow, oirow))| {
                    for l in 0..nao {
                        let (dr, di) = (dm.re[j * nao + l], dm.im[j * nao + l]);
                        let ab = l * ngrids;
                        for g in 0..ngrids {
                            let (ar, ai) = (ao.re[ab + g], ao.im[ab + g]);
                            orow[g] += dr * ar - di * ai;
                            oirow[g] += dr * ai + di * ar;
                        }
                    }
                });
            CTensor::from_planes(re, im)
        };
        let (h, host_ms) = timed(reps, host);
        let (d, dev_ms) = timed(reps, || {
            zgemm_dense(&client, &dm, &ao, nao, nao, ngrids).expect("zgemm")
        });
        reports.push(finish_contract(
            "(nao,nao).(nao,Ng)  [reduce over nao]",
            nao,
            ngrids,
            gflop,
            host_ms,
            dev_ms,
            &h,
            &d,
        ));
    }

    // ---- shape 2: (nao,Ng) . (Ng,nao), reduction over the GRID -------------
    {
        let gflop = 8.0 * (nao * nao * ngrids) as f64 * 1e-9;
        // The host route is D-PBC-17's ordered `oracle_dot`, which is what
        // `accumulate_vk` actually calls today — comparing against a naive `+=`
        // loop would be comparing against code that no longer exists.
        let host = || {
            let mut re = vec![0.0_f64; nao * nao];
            let mut im = vec![0.0_f64; nao * nao];
            re.par_chunks_mut(nao)
                .zip(im.par_chunks_mut(nao))
                .enumerate()
                .for_each(|(p, (vrow, virow))| {
                    let pb = p * ngrids;
                    let (xr, xi) = (&ao.re[pb..pb + ngrids], &ao.im[pb..pb + ngrids]);
                    for q in 0..nao {
                        let qb = q * ngrids;
                        let (yr, yi) = (&ao.re[qb..qb + ngrids], &ao.im[qb..qb + ngrids]);
                        vrow[q] = oracle_dot(xr, yr) - oracle_dot(xi, yi);
                        virow[q] = oracle_dot(xr, yi) + oracle_dot(xi, yr);
                    }
                });
            CTensor::from_planes(re, im)
        };
        // `zgemm_dense` wants `(nao,Ng) . (Ng,nao)`, so the right operand is the
        // transpose. Materialising it is part of the cost of this route and is
        // timed with it.
        let (h, host_ms) = timed(reps, host);
        let (d, dev_ms) = timed(reps, || {
            let mut tre = vec![0.0_f64; nao * ngrids];
            let mut tim = vec![0.0_f64; nao * ngrids];
            for q in 0..nao {
                for g in 0..ngrids {
                    tre[g * nao + q] = ao.re[q * ngrids + g];
                    tim[g * nao + q] = ao.im[q * ngrids + g];
                }
            }
            let t = CTensor::from_planes(tre, tim);
            zgemm_dense(&client, &ao, &t, nao, ngrids, nao).expect("zgemm")
        });
        reports.push(finish_contract(
            "(nao,Ng).(Ng,nao)   [reduce over GRID]",
            nao,
            ngrids,
            gflop,
            host_ms,
            dev_ms,
            &h,
            &d,
        ));
    }

    for r in &reports {
        println!(
            "{}\n  nao={} Ng={}  {:.3} GFLOP\n  \
             host (rayon, ordered) = {:8.3} ms  ({:6.2} GFLOP/s)\n  \
             zgemm_dense (device)  = {:8.3} ms  ({:6.2} GFLOP/s)   -> {:.2}x {}\n  \
             max |host - device|   = {:.3e}",
            r.shape,
            r.nao,
            r.ngrids,
            r.gflop,
            r.host_ms,
            r.host_gflops,
            r.device_zgemm_ms,
            r.device_gflops,
            (r.device_zgemm_ms / r.host_ms).max(r.host_ms / r.device_zgemm_ms),
            if r.device_zgemm_ms > r.host_ms {
                "SLOWER on the device"
            } else {
                "faster on the device"
            },
            r.max_abs_diff,
        );
    }

    if let Some(path) = &json_path {
        let s = serde_json::to_string_pretty(&reports).expect("serialize");
        std::fs::write(path, s).expect("write json");
        println!("wrote {path}");
    }
}

/// Best-of-`reps` wall time, so a scheduler hiccup cannot make either route look
/// bad. Returns the last result alongside it for the accuracy comparison.
fn timed<T>(reps: usize, mut f: impl FnMut() -> T) -> (T, f64) {
    let mut best = f64::INFINITY;
    let mut out = f();
    for _ in 0..reps {
        let (v, ms) = time_ms(&mut f);
        best = best.min(ms);
        out = v;
    }
    (out, best)
}

#[allow(clippy::too_many_arguments)]
fn finish_contract(
    shape: &str,
    nao: usize,
    ngrids: usize,
    gflop: f64,
    host_ms: f64,
    device_zgemm_ms: f64,
    h: &pyscf_algebra::CTensor,
    d: &pyscf_algebra::CTensor,
) -> ContractReport {
    let mut max_abs_diff = 0.0_f64;
    for i in 0..h.len().min(d.len()) {
        max_abs_diff = max_abs_diff
            .max((h.re[i] - d.re[i]).abs())
            .max((h.im[i] - d.im[i]).abs());
    }
    ContractReport {
        shape: shape.to_string(),
        nao,
        ngrids,
        gflop,
        host_ms,
        device_zgemm_ms,
        host_gflops: gflop / (host_ms * 1e-3),
        device_gflops: gflop / (device_zgemm_ms * 1e-3),
        max_abs_diff,
    }
}

// ---------------------------------------------------------------------------
// `ksymm` — S-00 of `.planning/pbc/KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN.md`.
//
// Phase 17 shipped `KsymAdaptedKrks` / `KsymAdaptedKuks` and 17-08 Task 5's
// "speed report" was never written, so **no k-symmetry timing has ever been
// measured in this repository**. RULE O forbids landing S-01..S-03 without a
// baseline; this is that baseline.
//
// What it measures, and why each number is the honest one:
//
// * `kernel()` for the ksymm driver and for the ordinary full-BZ driver on the
//   SAME cell, mesh and functional. Two SCFs, so the comparison is end to end.
//   It is a SPEED comparison only — RULE K forbids reading the two energies
//   against each other without first establishing that the full-BZ solution is
//   star-symmetric, which this harness does not do and does not claim.
// * `get_veff` warm on the converged density, both drivers — the per-iteration
//   number, which is what an optimisation item actually moves.
// * `unfold_kdms` alone, on the same density: S-01's target, isolated.
// * Peak RSS (`VmHWM`), because half of this plan is about memory and a table
//   without it reads as "never measured".
// ---------------------------------------------------------------------------

/// Peak resident set size in MiB, from `/proc/self/status`'s `VmHWM`.
///
/// `VmHWM` is a HIGH-WATER MARK and never decreases, so it is read once at the
/// end and reported as the run's peak — not differenced across stages, which
/// would be meaningless for a monotone counter.
fn peak_rss_mib() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmHWM:")).and_then(|l| {
                l.split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse::<f64>().ok())
            })
        })
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

/// The 1-minute load average, printed with every report: RULE O invalidates a
/// ratio measured on a contended machine, so the reader must be able to see
/// whether this one was.
fn load_average() -> f64 {
    std::fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(f64::NAN)
}

/// RULE O: measurements written into a `baselines` directory are evidence,
/// not scratch output.  Refuse to create them while the host is contended so
/// an invalid timing row cannot accidentally become a checked-in baseline.
fn guard_idle_baseline_write(path: &str, load: f64) {
    let is_baseline = std::path::Path::new(path)
        .components()
        .any(|component| component.as_os_str() == "baselines");
    assert!(
        !is_baseline || load <= 4.0,
        "refusing to write baseline {path:?}: 1-minute load average {load:.2} exceeds 4.0 (RULE O)"
    );
}

#[derive(Serialize, Default)]
struct KsymmReport {
    cell: String,
    basis: String,
    nk: [usize; 3],
    mesh: [usize; 3],
    xc: String,
    hybrid: bool,
    driver: String,
    nao: usize,
    nkpts: usize,
    nkpts_ibz: usize,
    /// `nkpts / nkpts_ibz` — the naive bound on any IBZ saving.
    fold_factor: f64,
    /// The star multiplicities, as weights. Printed because a uniform vector
    /// would mean the fixture cannot see a dropped weight at all
    /// (`krks_ksymm.rs::si_222_stars_have_unequal_sizes_so_the_weighting_is_observable`).
    weights_ibz: Vec<f64>,
    use_ao_symmetry: bool,
    /// End-to-end SCF, full BZ and IBZ.
    full_kernel_ms: f64,
    ksymm_kernel_ms: f64,
    ksymm_over_full_kernel: f64,
    /// Warm `get_veff` on the converged density — the per-iteration cost.
    warm_full_get_veff_ms: f64,
    warm_ksymm_get_veff_ms: f64,
    ksymm_over_full_get_veff: f64,
    /// S-01's target, isolated: one IBZ-to-BZ density unfold.
    warm_unfold_kdms_ms: f64,
    /// How many unfolds one `get_veff` is worth, at this cell.
    unfolds_per_get_veff_equivalent: f64,
    /// S-07: cold `eval_ao_kpts` calls inside each `kernel()` — count, summed
    /// ms, and the k-point count of every call.
    full_ao_calls: u64,
    full_ao_ms: f64,
    full_ao_nkpts: Vec<u64>,
    ksymm_ao_calls: u64,
    ksymm_ao_ms: f64,
    ksymm_ao_nkpts: Vec<u64>,
    peak_rss_mib: f64,
    load_average_at_start: f64,
    full_e_tot: f64,
    ksymm_e_tot: f64,
    full_converged: bool,
    ksymm_converged: bool,
}

#[derive(Serialize)]
struct KsymmJkRouteReport {
    cell: String,
    basis: String,
    nk: [usize; 3],
    mesh: [usize; 3],
    df: String,
    requested_route: String,
    nkpts: usize,
    nkpts_ibz: usize,
    reference_ms: f64,
    band_ms: f64,
    reference_over_band: f64,
    max_abs_delta: f64,
    bit_exact: bool,
    e_tot: f64,
    converged: bool,
    peak_rss_mib: f64,
    load_average_at_start: f64,
}

fn write_json_report<T: Serialize>(path: Option<String>, report: &T) {
    if let Some(path) = path {
        std::fs::write(
            &path,
            serde_json::to_string_pretty(report).expect("serialize"),
        )
        .expect("write json");
        println!("wrote {path}");
    }
}

fn run_ksymm_jk_route(
    args: &[String],
    cell_name: String,
    basis: String,
    nk: [usize; 3],
    mesh: [usize; 3],
    mut cell: Cell,
    kpts: pyscf_pbc_symm::kpts::KPoints,
    load0: f64,
    json_path: Option<String>,
) {
    use pyscf_pbc_scf::khf_ksymm::{JkRoute, KsymAdaptedKrhf};
    use pyscf_pbc_scf::khooks::KOverrideHooks;

    cell.mesh = mesh;
    let df_name = arg_value(args, "--df").unwrap_or_else(|| "fftdf".into());
    let requested_route = arg_value(args, "--jk-route").unwrap_or_else(|| "both".into());
    assert!(
        matches!(requested_route.as_str(), "reference" | "band" | "both"),
        "--jk-route must be reference, band, or both, got {requested_route:?}"
    );
    let mut with_df: Box<dyn PeriodicDf> = match df_name.as_str() {
        "fftdf" => Box::new(Fftdf::with_mesh(cell.clone(), &kpts.kpts, mesh).expect("FFTDF")),
        "gdf" => Box::new(Gdf::new(cell.clone(), &kpts.kpts)),
        _ => panic!("--df must be fftdf or gdf, got {df_name:?}"),
    };
    // Make lazy setup explicit and exclude it from both route timings.
    with_df.build().expect("density-fitting build");
    let mut mf = KsymAdaptedKrhf::from_df(with_df, kpts.clone());
    mf.use_ao_symmetry = !args.iter().any(|a| a == "--no-ao-symmetry");
    let cfg = KScfConfig {
        conv_tol: 1e-9,
        max_cycle: 40,
        ..KScfConfig::default()
    };
    mf.jk_route = JkRoute::Reference;
    let result = mf.kernel(&cfg).expect("reference-route KRHF kernel");

    // Prime both paths before timing. In particular GDF may lazily materialise
    // band-pair data on the first band call.
    mf.jk_route = JkRoute::Reference;
    let reference = mf.get_veff(&result.dm).expect("reference get_veff warmup");
    mf.jk_route = JkRoute::Band;
    let band = mf.get_veff(&result.dm).expect("band get_veff warmup");

    mf.jk_route = JkRoute::Reference;
    let (_, reference_ms) = time_ms(|| mf.get_veff(&result.dm).expect("reference get_veff"));
    mf.jk_route = JkRoute::Band;
    let (_, band_ms) = time_ms(|| mf.get_veff(&result.dm).expect("band get_veff"));

    let mut max_abs_delta = 0.0_f64;
    let mut bit_exact = true;
    for (a_set, b_set) in reference.iter().zip(&band) {
        for (a, b) in a_set.iter().zip(b_set) {
            for ((ar, ai), (br, bi)) in a.re.iter().zip(&a.im).zip(b.re.iter().zip(&b.im)) {
                max_abs_delta = max_abs_delta.max((ar - br).abs()).max((ai - bi).abs());
                bit_exact &= ar.to_bits() == br.to_bits() && ai.to_bits() == bi.to_bits();
            }
        }
    }
    let report = KsymmJkRouteReport {
        cell: cell_name,
        basis,
        nk,
        mesh,
        df: df_name,
        requested_route,
        nkpts: kpts.nkpts(),
        nkpts_ibz: kpts.nkpts_ibz(),
        reference_ms,
        band_ms,
        reference_over_band: reference_ms / band_ms,
        max_abs_delta,
        bit_exact,
        e_tot: result.e_tot,
        converged: result.converged,
        peak_rss_mib: peak_rss_mib(),
        load_average_at_start: load0,
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize")
    );
    write_json_report(json_path, &report);
}

fn run_ksymm(args: &[String]) {
    use pyscf_pbc_dft::krks_ksymm::{KsymAdaptedKrks, KsymAdaptedKuks};
    // `cell()` is a `KOverrideHooks` method on both adapters, not an inherent
    // one — the trait has to be in scope to call it.
    use pyscf_pbc_scf::khooks::KOverrideHooks;
    use pyscf_pbc_symm::kpts::make_kpts;

    let cell_name = arg_value(args, "--cell").unwrap_or_else(|| "si".into());
    let basis = arg_value(args, "--basis").unwrap_or_else(|| "gth-szv".into());
    let nk = parse_triple(&arg_value(args, "--nk").unwrap_or_else(|| "2,2,2".into()));
    let mesh = parse_triple(&arg_value(args, "--mesh").unwrap_or_else(|| "31,31,31".into()));
    let xc = arg_value(args, "--xc").unwrap_or_else(|| "pbe".into());
    let driver = arg_value(args, "--driver").unwrap_or_else(|| "krks".into());
    assert!(
        matches!(driver.as_str(), "krks" | "kuks" | "krhf-ksymm"),
        "--driver must be krks, kuks, or krhf-ksymm, got {driver:?}"
    );
    // D-17-07-01 (`17-07-SUMMARY.md`): `little_cogroup_ops` indexes `k2opk`'s
    // doubled column space while its consumers index `ops`, so time reversal
    // combined with `use_ao_symmetry = true` is refused by this port (and
    // raises `IndexError` upstream). The default here folds on the space group
    // alone, exactly as every ksymm test in the tree does.
    let time_reversal = args.iter().any(|a| a == "--time-reversal");
    let use_ao_symmetry = !args.iter().any(|a| a == "--no-ao-symmetry");
    let json_path = arg_value(args, "--json");
    let compare_path = arg_value(args, "--compare");

    let load0 = load_average();
    if let Some(path) = json_path.as_deref() {
        guard_idle_baseline_write(path, load0);
    }
    let mut cell = cell_by_name(&cell_name, &basis);
    cell.mesh = mesh;
    let kpts_abs = make_kpts_default(&cell, nk).expect("k-mesh");
    let kpts = make_kpts(&cell, &kpts_abs, true, time_reversal).expect("make_kpts");
    assert!(
        kpts.nkpts_ibz() < kpts.nkpts(),
        "this cell/k-mesh does not fold ({} IBZ of {} BZ) — nothing to measure",
        kpts.nkpts_ibz(),
        kpts.nkpts()
    );
    if use_ao_symmetry {
        use pyscf_pbc_symm::basis::{self, SymmAdaptedBasisInput};
        let input = SymmAdaptedBasisInput {
            kpts_scaled_ibz: kpts.kpts_scaled_ibz.clone(),
            little_cogroup_ops: kpts.little_cogroup_ops.clone(),
            ops: kpts.symmetry.ops.clone(),
            dmats: kpts.symmetry.dmats.clone(),
        };
        basis::build_symmetry(&mut cell, &input).expect("build_symmetry");
    }

    if driver == "krhf-ksymm" {
        return run_ksymm_jk_route(
            args, cell_name, basis, nk, mesh, cell, kpts, load0, json_path,
        );
    }

    let cfg = KScfConfig {
        conv_tol: 1e-9,
        max_cycle: 40,
        ..KScfConfig::default()
    };
    let is_hybrid = hybrid(&xc).unwrap_or(false);
    let nao = cell.mol.nao_nr;

    // ---- the ordinary full-BZ driver ----
    let full_ao = AoCallLayer::default();
    let ksymm_ao = AoCallLayer::default();
    let df_full = Fftdf::with_mesh(cell.clone(), &kpts.kpts, mesh).expect("FFTDF");
    let (full_result, full_kernel_ms, warm_full_get_veff_ms) = if driver == "kuks" {
        let mf = Kuks::from_df(Box::new(df_full), &xc).expect("KUKS");
        let (r, ms) = tracing::subscriber::with_default(
            tracing_subscriber::registry().with(full_ao.clone()),
            || time_ms(|| mf.kernel(&cfg).expect("KUKS kernel")),
        );
        // Warm: the AO / coulG caches are already populated by the SCF above,
        // so this is what one converged iteration actually costs.
        let (_, veff_ms) = time_ms(|| mf.get_veff_tagged(&r.dm, None).expect("get_veff"));
        (r, ms, veff_ms)
    } else {
        let mf = Krks::from_df(Box::new(df_full), &xc).expect("KRKS");
        let (r, ms) = tracing::subscriber::with_default(
            tracing_subscriber::registry().with(full_ao.clone()),
            || time_ms(|| mf.kernel(&cfg).expect("KRKS kernel")),
        );
        let (_, veff_ms) = time_ms(|| mf.get_veff_tagged(&r.dm, None).expect("get_veff"));
        (r, ms, veff_ms)
    };

    // ---- the k-symmetric driver ----
    let df_sym = Fftdf::with_mesh(cell.clone(), &kpts.kpts, mesh).expect("FFTDF");
    let grids = pyscf_pbc_dft::gen_grid::PeriodicGrids::uniform(&cell, Some(mesh)).expect("grids");
    let (ksymm_result, ksymm_kernel_ms, warm_ksymm_get_veff_ms, warm_unfold_kdms_ms) = if driver
        == "kuks"
    {
        let mut mf =
            KsymAdaptedKuks::new(cell.clone(), kpts.clone(), &xc).expect("KsymAdaptedKuks");
        mf.with_df = Box::new(df_sym);
        mf.grids = grids;
        mf.use_ao_symmetry = use_ao_symmetry;
        let (r, ms) = tracing::subscriber::with_default(
            tracing_subscriber::registry().with(ksymm_ao.clone()),
            || time_ms(|| mf.kernel(&cfg).expect("ksymm KUKS kernel")),
        );
        let (_, veff_ms) = time_ms(|| mf.get_veff_tagged(&r.dm, None).expect("get_veff"));
        let (_, unfold_ms) = time_ms(|| mf.ni.unfold_kdms(mf.cell(), &r.dm, nao).expect("unfold"));
        (r, ms, veff_ms, unfold_ms)
    } else {
        let mut mf =
            KsymAdaptedKrks::new(cell.clone(), kpts.clone(), &xc).expect("KsymAdaptedKrks");
        mf.with_df = Box::new(df_sym);
        mf.grids = grids;
        mf.use_ao_symmetry = use_ao_symmetry;
        let (r, ms) = tracing::subscriber::with_default(
            tracing_subscriber::registry().with(ksymm_ao.clone()),
            || time_ms(|| mf.kernel(&cfg).expect("ksymm KRKS kernel")),
        );
        let (_, veff_ms) = time_ms(|| mf.get_veff_tagged(&r.dm, None).expect("get_veff"));
        let (_, unfold_ms) = time_ms(|| mf.ni.unfold_kdms(mf.cell(), &r.dm, nao).expect("unfold"));
        (r, ms, veff_ms, unfold_ms)
    };

    let report = KsymmReport {
        cell: cell_name,
        basis,
        nk,
        mesh,
        xc: xc.clone(),
        hybrid: is_hybrid,
        driver: driver.clone(),
        nao,
        nkpts: kpts.nkpts(),
        nkpts_ibz: kpts.nkpts_ibz(),
        fold_factor: kpts.nkpts() as f64 / kpts.nkpts_ibz() as f64,
        weights_ibz: kpts.weights_ibz.clone(),
        use_ao_symmetry,
        full_kernel_ms,
        ksymm_kernel_ms,
        ksymm_over_full_kernel: ksymm_kernel_ms / full_kernel_ms,
        warm_full_get_veff_ms,
        warm_ksymm_get_veff_ms,
        ksymm_over_full_get_veff: warm_ksymm_get_veff_ms / warm_full_get_veff_ms,
        warm_unfold_kdms_ms,
        unfolds_per_get_veff_equivalent: warm_ksymm_get_veff_ms / warm_unfold_kdms_ms.max(1e-12),
        full_ao_calls: full_ao.summary().0,
        full_ao_ms: full_ao.summary().1,
        full_ao_nkpts: full_ao.summary().2,
        ksymm_ao_calls: ksymm_ao.summary().0,
        ksymm_ao_ms: ksymm_ao.summary().1,
        ksymm_ao_nkpts: ksymm_ao.summary().2,
        peak_rss_mib: peak_rss_mib(),
        load_average_at_start: load0,
        full_e_tot: full_result.e_tot,
        ksymm_e_tot: ksymm_result.e_tot,
        full_converged: full_result.converged,
        ksymm_converged: ksymm_result.converged,
    };

    println!(
        "cell={} basis={} nk={:?} mesh={:?} xc={} hybrid={} driver={}\n  \
         nao={} nkpts={} nkpts_ibz={} fold={:.3}x  use_ao_symmetry={}\n  \
         weights_ibz={:?}\n  \
         load average at start = {:.2}   peak RSS = {:.1} MiB\n  \
         kernel()   full = {:.3} ms   ksymm = {:.3} ms   ksymm/full = {:.4}\n  \
         get_veff   full = {:.3} ms   ksymm = {:.3} ms   ksymm/full = {:.4}\n  \
         unfold_kdms (one call) = {:.3} ms   -> {:.1} unfolds per ksymm get_veff\n  \
         cold eval_ao_kpts inside kernel(): full {} call(s) = {:.1} ms at nkpts {:?}; \
         ksymm {} call(s) = {:.1} ms at nkpts {:?}\n  \
         e_tot      full = {:.12} ({})   ksymm = {:.12} ({})\n  \
         NOTE: the two energies are NOT a correctness comparison — RULE K. An\n  \
         IBZ run is CONSTRAINED to symmetric occupations and an unconstrained\n  \
         full-BZ run is not, so they can legitimately converge to different\n  \
         states (D-17-08-03). This subcommand measures TIME.",
        report.cell,
        report.basis,
        report.nk,
        report.mesh,
        report.xc,
        report.hybrid,
        report.driver,
        report.nao,
        report.nkpts,
        report.nkpts_ibz,
        report.fold_factor,
        report.use_ao_symmetry,
        report.weights_ibz,
        report.load_average_at_start,
        report.peak_rss_mib,
        report.full_kernel_ms,
        report.ksymm_kernel_ms,
        report.ksymm_over_full_kernel,
        report.warm_full_get_veff_ms,
        report.warm_ksymm_get_veff_ms,
        report.ksymm_over_full_get_veff,
        report.warm_unfold_kdms_ms,
        report.unfolds_per_get_veff_equivalent,
        report.full_ao_calls,
        report.full_ao_ms,
        report.full_ao_nkpts,
        report.ksymm_ao_calls,
        report.ksymm_ao_ms,
        report.ksymm_ao_nkpts,
        report.full_e_tot,
        report.full_converged,
        report.ksymm_e_tot,
        report.ksymm_converged,
    );

    if let Some(path) = json_path {
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&report).expect("serialize"),
        )
        .expect("write json");
        println!("wrote {path}");
    }
    if let Some(path) = compare_path {
        let prev: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read baseline"))
                .expect("parse baseline");
        let now = serde_json::to_value(&report).expect("serialize");
        println!("--- compare against {path} (RULE O: ONE variable changed) ---");
        for key in [
            "full_kernel_ms",
            "ksymm_kernel_ms",
            "ksymm_over_full_kernel",
            "warm_full_get_veff_ms",
            "warm_ksymm_get_veff_ms",
            "ksymm_over_full_get_veff",
            "warm_unfold_kdms_ms",
            "full_ao_ms",
            "ksymm_ao_ms",
            "peak_rss_mib",
        ] {
            let (a, b) = (
                prev.get(key).and_then(|v| v.as_f64()),
                now.get(key).and_then(|v| v.as_f64()),
            );
            if let (Some(a), Some(b)) = (a, b) {
                println!(
                    "  {key:<28} {a:>12.3} -> {b:>12.3}   ({:+.1} %)",
                    (b / a - 1.0) * 100.0
                );
            }
        }
    }
}

#[derive(Serialize)]
struct AoReport {
    driver: String,
    rho_route: String,
    cell: String,
    basis: String,
    nk: [usize; 3],
    mesh: [usize; 3],
    deriv: usize,
    screen: bool,
    nao: usize,
    nkpts: usize,
    ngrids: usize,
    nimgs: usize,
    /// Images the W-09 screen let through to a kernel launch.
    launched_images: u64,
    /// Grid points those launches covered, summed — `launched_images ·
    /// ngrids` when the screen is off.
    kept_points_total: u64,
    /// `kept_points_total / (nimgs · ngrids)`.
    kept_fraction: f64,
    comp: usize,
    ao_block_reals: usize,
    ao_cache_bytes: usize,
    legacy_round_trip_bytes_per_image: usize,
    legacy_zero_fill_bytes_per_image: usize,
    round_trip_bytes_per_image: usize,
    zero_fill_bytes_per_image: usize,
    phase_upload_bytes_per_image: usize,
    cold_eval_ao_kpts_ms: f64,
    shift_pack_ms: f64,
    eval_gto_ms: f64,
    scatter_ms: f64,
    k08_accumulate_ms: f64,
    peak_rss_mib: f64,
    load_average_at_start: f64,
}

fn run_ao(args: &[String]) {
    use pyscf_pbc_dft::gen_grid::PeriodicGrids;
    use pyscf_pbc_gto::{estimate_rcut_for_eval, eval_ao_kpts, lattice};

    let cell_name = arg_value(args, "--cell").unwrap_or_else(|| "si".into());
    let driver = arg_value(args, "--driver").unwrap_or_else(|| "full".into());
    let rho_route = arg_value(args, "--rho-route").unwrap_or_else(|| "unfold".into());
    let basis = arg_value(args, "--basis").unwrap_or_else(|| "gth-szv".into());
    let nk = parse_triple(&arg_value(args, "--nk").unwrap_or_else(|| "2,2,2".into()));
    let mesh = parse_triple(&arg_value(args, "--mesh").unwrap_or_else(|| "31,31,31".into()));
    let deriv: usize = arg_value(args, "--deriv")
        .unwrap_or_else(|| "0".into())
        .parse()
        .expect("--deriv must be 0 or 1");
    assert!(deriv <= 1, "--deriv must be 0 or 1");
    let screen_arg = arg_value(args, "--screen").unwrap_or_else(|| "on".into());
    let screen = match screen_arg.as_str() {
        "on" => true,
        "off" => false,
        _ => panic!("--screen must be on or off, got {screen_arg:?}"),
    };
    // The screen setting is process-global and cached on first use.  This
    // profiler performs exactly one AO run per process, so setting it here is
    // deterministic and mirrors the public kill-switch contract.
    unsafe { std::env::set_var("PYSCF_PBC_AO_SCREEN", if screen { "1" } else { "0" }) };

    let load0 = load_average();
    let json_path = arg_value(args, "--json");
    if let Some(path) = json_path.as_deref() {
        guard_idle_baseline_write(path, load0);
    }
    let compare_path = arg_value(args, "--compare");

    let mut cell = cell_by_name(&cell_name, &basis);
    cell.mesh = mesh;
    let kpts_full = make_kpts_default(&cell, nk).expect("k-mesh");
    let kpts = if driver == "kuks-ksymm" && rho_route == "symmetrize" {
        pyscf_pbc_symm::kpts::make_kpts(&cell, &kpts_full, true, false)
            .expect("symmetric k-mesh")
            .kpts_ibz
    } else {
        kpts_full
    };
    let grids = PeriodicGrids::uniform(&cell, Some(mesh)).expect("uniform grid");
    let coords = grids.coords().expect("uniform coordinates");
    let eval_name = if deriv == 0 {
        "GTOval_sph"
    } else {
        "GTOval_sph_deriv1"
    };
    let rcut = estimate_rcut_for_eval(&cell, deriv as u32).expect("AO rcut");
    let rmax = rcut.iter().copied().fold(0.0_f64, f64::max);
    let nimgs = lattice::get_lattice_ls(&cell, Some(rmax), None, false)
        .expect("AO image list")
        .len();

    let stages = AoStageLayer::default();
    let subscriber = tracing_subscriber::registry().with(stages.clone());
    let (out, cold_eval_ao_kpts_ms) = tracing::subscriber::with_default(subscriber, || {
        time_ms(|| eval_ao_kpts(&cell, eval_name, coords, &kpts).expect("eval_ao_kpts"))
    });
    let [shift_pack_ms, eval_gto_ms, scatter_ms, k08_accumulate_ms] = stages.milliseconds();
    let (launched_images, kept_points_total) = stages.launches();
    let n = out.comp * out.ngrids * out.nao;
    let report = AoReport {
        driver,
        rho_route,
        cell: cell_name,
        basis,
        nk,
        mesh,
        deriv,
        screen,
        nao: out.nao,
        nkpts: out.nkpts(),
        ngrids: out.ngrids,
        nimgs,
        launched_images,
        kept_points_total,
        kept_fraction: kept_points_total as f64 / (nimgs * out.ngrids).max(1) as f64,
        comp: out.comp,
        ao_block_reals: n,
        ao_cache_bytes: out.nkpts() * n * 2 * size_of::<f64>(),
        legacy_round_trip_bytes_per_image: 2 * n * size_of::<f64>(),
        legacy_zero_fill_bytes_per_image: n * size_of::<f64>(),
        round_trip_bytes_per_image: 0,
        zero_fill_bytes_per_image: 0,
        phase_upload_bytes_per_image: 2 * out.nkpts() * size_of::<f64>(),
        cold_eval_ao_kpts_ms,
        shift_pack_ms,
        eval_gto_ms,
        scatter_ms,
        k08_accumulate_ms,
        peak_rss_mib: peak_rss_mib(),
        load_average_at_start: load0,
    };
    println!(
        "driver={} rho_route={} cell={} basis={} nk={:?} mesh={:?} deriv={} screen={}\n  \
         nao={} nkpts={} ngrids={} nimgs={} launched={} kept_points={} ({:.3} of nimgs*ngrids) \
         comp={} n={} cache={} bytes\n  \
         RULE T/image: legacy round-trip={} bytes + zero-fill={} bytes; \
         resident round-trip={} bytes + zero-fill={} bytes + phases={} bytes\n  \
         load average at start={:.2} peak RSS={:.1} MiB\n  \
         eval_ao_kpts cold={:.3} ms\n  \
         shift+pack={:.3} ms ({:.1}%) eval_gto={:.3} ms ({:.1}%)\n  \
         scatter={:.3} ms ({:.1}%) k08_accumulate={:.3} ms ({:.1}%)",
        report.driver,
        report.rho_route,
        report.cell,
        report.basis,
        report.nk,
        report.mesh,
        report.deriv,
        report.screen,
        report.nao,
        report.nkpts,
        report.ngrids,
        report.nimgs,
        report.launched_images,
        report.kept_points_total,
        report.kept_fraction,
        report.comp,
        report.ao_block_reals,
        report.ao_cache_bytes,
        report.legacy_round_trip_bytes_per_image,
        report.legacy_zero_fill_bytes_per_image,
        report.round_trip_bytes_per_image,
        report.zero_fill_bytes_per_image,
        report.phase_upload_bytes_per_image,
        report.load_average_at_start,
        report.peak_rss_mib,
        report.cold_eval_ao_kpts_ms,
        report.shift_pack_ms,
        100.0 * report.shift_pack_ms / report.cold_eval_ao_kpts_ms,
        report.eval_gto_ms,
        100.0 * report.eval_gto_ms / report.cold_eval_ao_kpts_ms,
        report.scatter_ms,
        100.0 * report.scatter_ms / report.cold_eval_ao_kpts_ms,
        report.k08_accumulate_ms,
        100.0 * report.k08_accumulate_ms / report.cold_eval_ao_kpts_ms,
    );

    if let Some(path) = json_path {
        std::fs::write(
            &path,
            serde_json::to_string_pretty(&report).expect("serialize"),
        )
        .expect("write json");
        println!("wrote {path}");
    }
    if let Some(path) = compare_path {
        let prev: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&path).expect("read baseline"))
                .expect("parse baseline");
        let now = serde_json::to_value(&report).expect("serialize");
        println!("--- compare against {path} (RULE O: ONE variable changed) ---");
        for key in [
            "cold_eval_ao_kpts_ms",
            "shift_pack_ms",
            "eval_gto_ms",
            "scatter_ms",
            "k08_accumulate_ms",
            "peak_rss_mib",
        ] {
            let (a, b) = (
                prev.get(key).and_then(|v| v.as_f64()),
                now.get(key).and_then(|v| v.as_f64()),
            );
            if let (Some(a), Some(b)) = (a, b) {
                println!(
                    "  {key:<28} {a:>12.3} -> {b:>12.3}   ({:+.1} %)",
                    (b / a - 1.0) * 100.0
                );
            }
        }
    }
}

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn parse_triple(s: &str) -> [usize; 3] {
    let parts: Vec<usize> = s
        .split(',')
        .map(|t| t.trim().parse().expect("integer"))
        .collect();
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
        "contract" => run_contract(rest),
        "ksymm" => run_ksymm(rest),
        "ao" => run_ao(rest),
        "multigrid" => run_multigrid(rest),
        other => {
            eprintln!(
                "unknown subcommand {other:?} (expected transform|jk|contract|ksymm|ao|multigrid)"
            );
            std::process::exit(1);
        }
    }
}

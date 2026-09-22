//! Where `get_hcore`'s local half actually spends its peak RSS.
//!
//! Not a pass/fail gate — an INSTRUMENT, and it is `#[ignore]`d so a normal
//! `cargo test` never pays for it. The `hcore` profile mode reported the two
//! routes at the same peak, which is only interesting if you can say which
//! stage set that peak. Each probe below runs in its own process (one `--exact`
//! invocation each) because CubeCL pools device buffers per process-global
//! client: a second probe in the same process starts with whatever the first
//! one left in the pool and its `VmHWM` reads high.
//!
//! ```bash
//! for t in ao_table_only host_route fused_route accumulator_only; do
//!   cargo test -p pyscf-pbc-df --release --test hcore_peak_rss -- \
//!     --ignored --nocapture --exact $t
//! done
//! ```

mod common;

use pyscf_pbc_df::{Fftdf, PeriodicDf};
use pyscf_pbc_gto::make_kpts_default;

/// Default shape. `PROBE_MESH=41` and `PROBE_NK=3` scale it up, so one build
/// serves the whole size sweep — the saving is a multiple of the AO table, and
/// a single size cannot show that.
const MESH: [usize; 3] = [31, 31, 31];
const NK: [usize; 3] = [2, 2, 2];

fn triple(var: &str, default: [usize; 3]) -> [usize; 3] {
    match std::env::var(var)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(n) if n > 0 => [n; 3],
        _ => default,
    }
}

fn status_kb(field: &str) -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with(field)).and_then(|l| {
                l.split_whitespace()
                    .nth(1)
                    .and_then(|v| v.parse::<f64>().ok())
            })
        })
        .unwrap_or(f64::NAN)
}

fn peak_mib() -> f64 {
    status_kb("VmHWM:") / 1024.0
}
fn rss_mib() -> f64 {
    status_kb("VmRSS:") / 1024.0
}

/// Reset `VmHWM` to the current `VmRSS` (Linux >= 4.0).
fn reset_peak() -> bool {
    std::fs::write("/proc/self/clear_refs", "5\n").is_ok()
}

struct Probe {
    label: &'static str,
    before: f64,
    table_mib: f64,
}

impl Probe {
    fn start(label: &'static str, nkpts: usize, nao: usize, ngrids: usize) -> Self {
        assert!(reset_peak(), "this kernel refuses /proc/self/clear_refs");
        let before = rss_mib();
        Self {
            label,
            before,
            table_mib: 16.0 * (nkpts * nao * ngrids) as f64 / (1024.0 * 1024.0),
        }
    }
    fn finish(self) {
        let (peak, after) = (peak_mib(), rss_mib());
        println!(
            "\n  {:<18} RSS before {:>8.1} | peak {:>8.1} | after {:>8.1} MiB\n  \
             {:<18} peak above baseline = {:>8.1} MiB = {:.2} x one AO table ({:.1} MiB)",
            self.label,
            self.before,
            peak,
            after,
            "",
            peak - self.before,
            (peak - self.before) / self.table_mib,
            self.table_mib,
        );
    }
}

fn setup() -> (pyscf_pbc_gto::Cell, Vec<[f64; 3]>, [usize; 3]) {
    let cell = common::diamond();
    let kpts = make_kpts_default(&cell, triple("PROBE_NK", NK)).expect("k-mesh");
    (cell, kpts, triple("PROBE_MESH", MESH))
}

/// The AO evaluation and read-back ALONE — `eval_ao_kpts` through the FFTDF
/// cache. Whatever this costs, both `get_hcore` routes inherit.
#[test]
#[ignore = "instrument, not a gate"]
fn ao_table_only() {
    let (cell, kpts, mesh) = setup();
    let df = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("FFTDF");
    let p = Probe::start("ao_table_only", kpts.len(), cell.mol.nao_nr, df.ngrids());
    let ao = df.ao_kpts(&kpts).expect("ao_kpts");
    std::hint::black_box(&ao);
    p.finish();
}

/// `get_pp` with the table materialised and reduced on the host.
#[test]
#[ignore = "instrument, not a gate"]
fn host_route() {
    let (cell, kpts, mesh) = setup();
    // SAFETY-equivalent note: one probe per process, set before any call reads it.
    unsafe { std::env::set_var("PYSCF_PBC_HCORE_FUSE", "0") };
    let df = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("FFTDF");
    let p = Probe::start("host_route", kpts.len(), cell.mol.nao_nr, df.ngrids());
    let v = df.get_pp(&kpts).expect("get_pp");
    std::hint::black_box(&v);
    p.finish();
}

/// `get_pp` with the contraction fused onto the device (K-14f).
#[test]
#[ignore = "instrument, not a gate"]
fn fused_route() {
    let (cell, kpts, mesh) = setup();
    unsafe { std::env::set_var("PYSCF_PBC_HCORE_FUSE", "1") };
    let df = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("FFTDF");
    let p = Probe::start("fused_route", kpts.len(), cell.mol.nao_nr, df.ngrids());
    let v = df.get_pp(&kpts).expect("get_pp");
    std::hint::black_box(&v);
    p.finish();
}

/// The accumulator alone, EXECUTED — its two device planes, resident.
///
/// Building the accumulator only QUEUES its fill kernels: CubeCL launches are
/// lazy, and `client.empty` maps address space without faulting a page. An
/// earlier version of this probe stopped there and so measured reservations —
/// 9.4 MiB for 227 MiB of planes after B-01. This one forces execution by
/// contracting the planes with `local_vmat_resident`: the queued fills run and
/// every page is read, while what comes back is only `nkpts · nao²` numbers, so
/// no host copy of the table enters the measurement. Zero planes contract to a
/// zero matrix; the value is irrelevant, only the residency is measured.
#[test]
#[ignore = "instrument, not a gate"]
fn accumulator_only() {
    let (cell, kpts, mesh) = setup();
    let df = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("FFTDF");
    let (nkpts, nao, ngrids) = (kpts.len(), cell.mol.nao_nr, df.ngrids());
    let client = pyscf_algebra::select_backend().expect("backend").client;
    let vr = vec![0.0f64; ngrids];
    let gamma = vec![false; nkpts];
    let p = Probe::start("accumulator_only", nkpts, nao, ngrids);
    let acc = pyscf_kernels::pbc::AoKAccumulator::zeros(&client, nkpts, nao * ngrids);
    let v = pyscf_kernels::pbc::local_vmat_resident(&client, acc, &vr, nao, ngrids, &gamma)
        .expect("contract to force execution");
    std::hint::black_box(&v);
    p.finish();
}

// There is deliberately no `image_batch_only` probe. The K-09 image batch's
// slots are written only by the AO evaluator, so no probe can make them
// resident without running the evaluation that `ao_table_only` already
// measures; a probe that merely constructs the batch measures an unfaulted
// reservation (8.9 MiB for a 134.4 MiB batch) and was removed as misleading.
// B-02's route peaks, not a batch probe, are the evidence for the batch.

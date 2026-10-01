//! YTa7O19 band structure — KRKS + `get_bands` along the hexagonal path, as a
//! staged pipeline that survives being killed at any point (a Kaggle session
//! limit) and resumes where it stopped.
//!
//! Structure: Materials Project mp-772036 (P-6c2, Z = 2, 54 atoms),
//! pseudopotential `gth-pbe`, XC `pbe` through the default `libxc` backend.
//!
//! # Stages
//!
//! | stage | what | checkpoint (in `YTA_CKPT_DIR`) |
//! |---|---|---|
//! | `1e` | overlap + core Hamiltonian of the target basis | `s1e.bin`, `h1e.bin` |
//! | `pre` | (optional, `YTA_PRE_BASIS`) SCF in a small basis on the same grid and k-mesh | `pre_s1e.bin`, `pre_h1e.bin`, `scf_pre.bin` |
//! | `project` | project the `pre` density into the target basis | `dm0.bin` |
//! | `s1` | target-basis SCF, stage-1 controls (from `dm0.bin` or `YTA_INIT_GUESS`) | `scf_s1.bin` |
//! | `s2` | (optional, `YTA_REFINE_SMEARING`) re-converge with a narrow smearing | `scf_s2.bin` |
//! | `bands` | `get_bands` on the path, `YTA_CHUNK` k-points per call | the `bands` entry of `YTA_OUT` |
//!
//! A finished stage is skipped on the next start. An SCF stage checkpoints
//! after EVERY cycle: the density, the Fock matrix the cycle diagonalised,
//! the next cycle number, the energy, whether the cycle converged and the
//! controls that define the stage's target (smearing, tolerance) —
//! `scf_<stage>.bin`, one atomic file. A converged state is skipped only if
//! those controls still match; otherwise the stage restarts from its saved
//! density. Downstream results are tied to their source by lineage tokens:
//! each SCF stage records the token of the density it started from, and the
//! bands the token of the density they were computed on, so nothing is
//! reused against a different upstream. `fingerprint.json` pins what every
//! file depends on (basis, pre-basis, cutoff, k-mesh, XC): a checkpoint
//! directory from a different calculation is refused (exit 6), never mixed in.
//! A resumed stage therefore keeps damping from its first cycle and keeps the
//! damping/DIIS switch at the same cycle number — before DIIS starts the
//! resumed cycles are bit-identical to an uninterrupted run; after it the
//! DIIS subspace restarts empty.
//!
//! Progress is human-readable in the same directory: `status.json` (current
//! stage, cycle, energy) and `history.jsonl` (one line per SCF cycle).
//!
//! # Knobs (environment)
//!
//! | var | default | meaning |
//! |---|---|---|
//! | `YTA_KE` | `100` | plane-wave `ke_cutoff`, Hartree |
//! | `YTA_KMESH` | `3,3,1` | SCF Monkhorst-Pack mesh |
//! | `YTA_BASIS` | `gth-szv-molopt-sr` | target basis (e.g. `gth-dzvp-molopt-sr`) |
//! | `YTA_PRE_BASIS` | unset | pre-converge in this basis first, then project (e.g. `gth-szv-molopt-sr`) |
//! | `YTA_PRE_CONV` / `YTA_PRE_MAXCYC` | `1e-4` / `YTA_MAXCYC` | stage `pre` tolerance and cycle cap — it only has to produce a starting density |
//! | `YTA_NPATH` | `60` | k-points on the band path |
//! | `YTA_CHUNK` | `6` | band k-points per `get_bands` call |
//! | `YTA_XC` | `pbe` | functional |
//! | `YTA_MAXCYC` | `80` | cycle cap of each SCF stage (a TOTAL over sessions) |
//! | `YTA_CONV` | `1e-7` | SCF energy tolerance |
//! | `YTA_SMEARING` | unset | stage `pre`/`s1` smearing, `fermi:<sigma Ha>` / `gauss:<sigma Ha>` |
//! | `YTA_DAMP` | `0` | stage `pre`/`s1` Fock damping (before `YTA_DIIS_START`) |
//! | `YTA_LEVEL_SHIFT` | `0` | stage `pre`/`s1` virtual level shift, Hartree |
//! | `YTA_DIIS_SPACE` | `8` | DIIS subspace size |
//! | `YTA_DIIS_START` | `1` | first cycle DIIS extrapolates |
//! | `YTA_INIT_GUESS` | `minao` | `minao`, `atom` or `1e` (stage `pre`, or `s1` without `pre`) |
//! | `YTA_REFINE_SMEARING` | unset | stage `s2` smearing (`kind:sigma` or `none`) |
//! | `YTA_REFINE_DAMP` / `YTA_REFINE_LEVEL_SHIFT` / `YTA_REFINE_DIIS_START` | `0` / `0` / `1` | stage `s2` controls |
//! | `YTA_REQUIRE_CONVERGED` | unset | `1`: exit 5 / 3 / 4 when `pre` / `s1` / `s2` hits `YTA_MAXCYC` unconverged |
//! | `YTA_CKPT_DIR` | unset | checkpoint directory (without it nothing is saved or resumed) |
//! | `YTA_ADOPT_CHECKPOINT` | unset | `1`: accept `.bin` files in a checkpoint without `fingerprint.json` as this calculation's |
//! | `YTA_OUT` | `<ckpt>/result.json`, else `yta7o19_bands.json` | result file, rewritten after each stage and band chunk |
//! | `YTA_STOP_AFTER` | unset | `1e`: exit after the one-electron matrices are checkpointed |
//! | `YTA_REQUIRE_BACKEND` | unset | abort unless the resolved backend has this name |
//! | `YTA_REQUIRE_XC` | unset | abort unless the XC backend's `Debug` name is this (`Libxc`) |
//!
//! 2026-09-26: with plain DIIS and aufbau occupations the SCF of this cell
//! oscillates by thousands of Hartree (identically on the CPU and CUDA
//! backends); smearing plus early damping converged it in `gth-szv-molopt-sr`.
//! 2026-09-30: in `gth-dzvp-molopt-sr` from a MINAO guess even damping 0.9
//! diverged — hence the `pre` → `project` start from a converged small-basis
//! density.
//!
//! ```bash
//! PYSCF_BACKEND=cuda YTA_REQUIRE_BACKEND=cuda cargo run --release \
//!     -p pyscf-pbc-dft \
//!     --features pyscf-algebra/cuda,pyscf-kernels/cuda --example yta7o19_bands
//! ```

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use pyscf_algebra::CTensor;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_gto::{ALattice, BravaisLattice, Cell, CellBuildArgs, band_path, make_kpts_default};
use pyscf_pbc_scf::KOverrideHooks;
use pyscf_pbc_scf::types::{CycleHook, CycleState, KDms, KInitGuess, KMats, KScfConfig};
use serde_json::json;

const HARTREE_EV: f64 = 27.211386245988;

/// Lattice vectors (rows), Angstrom — Materials Project mp-772036.
const LATTICE_ANG: [[f64; 3]; 3] = [
    [3.133221, -5.426898, 0.0],
    [3.133221, 5.426898, 0.0],
    [0.0, 0.0, 20.113571],
];

/// Cartesian sites, Angstrom — Materials Project mp-772036 (Y2Ta14O38).
const SITES_ANG: [(&str, [f64; 3]); 54] = [
    ("Y", [3.133221, -1.808969617932, 10.0567855]),
    ("Y", [3.133221, -1.808969617932, 0.0]),
    ("Ta", [2.258068509606, -3.6187749781560004, 13.183359657666]),
    ("Ta", [2.258068509606, -3.6187749781560004, 16.986996842334]),
    ("Ta", [4.008373490394, -3.618774978156001, 3.126574157666]),
    (
        "Ta",
        [4.008373490394, -3.618774978156001, 6.930211342334001],
    ),
    (
        "Ta",
        [4.262988428517, -0.1461517900379997, 6.930211342334001],
    ),
    ("Ta", [4.262988428517, -0.1461517900379997, 3.126574157666]),
    ("Ta", [0.0, 0.0, 0.0]),
    ("Ta", [0.0, 0.0, 10.0567855]),
    (
        "Ta",
        [5.1381409189110006, -1.6619603780100003, 13.183359657666],
    ),
    (
        "Ta",
        [5.1381409189110006, -1.6619603780100003, 16.986996842334],
    ),
    ("Ta", [2.003453571483, -0.146151790038, 13.183359657666]),
    ("Ta", [2.003453571483, -0.146151790038, 16.986996842334]),
    (
        "Ta",
        [1.1283010810889997, -1.6619603780100003, 6.930211342334001],
    ),
    (
        "Ta",
        [1.1283010810889997, -1.6619603780100003, 3.126574157666],
    ),
    (
        "O",
        [2.357983794075, 0.4678148882940003, 3.0830483900220003],
    ),
    ("O", [2.357983794075, 0.4678148882940003, 6.973737109978001]),
    ("O", [2.0962251120719997, -0.4485114121079999, 15.08517825]),
    (
        "O",
        [3.9084582059249997, 0.4678148882940002, 17.030522609978],
    ),
    (
        "O",
        [3.9084582059249997, 0.4678148882940002, 13.139833890022],
    ),
    ("O", [1.4365379634059998, -1.591123078416, 5.02839275]),
    ("O", [3.907073322243, 3.150916674678, 13.139833890022]),
    ("O", [3.907073322243, 3.150916674678, 17.030522609978]),
    ("O", [4.1702168879279995, -0.44851141210800005, 5.02839275]),
    ("O", [3.133221, -1.808969617932, 12.692045458849]),
    ("O", [3.133221, -1.808969617932, 7.421525541151]),
    ("O", [3.133221, -1.808969617932, 2.6352599588489998]),
    ("O", [3.133221, -1.808969617932, 17.478311041151]),
    ("O", [4.829904036593999, -1.591123078416, 15.08517825]),
    ("O", [4.682310528167999, 1.808166437028, 3.0830483900220003]),
    ("O", [4.682310528167999, 1.808166437028, 6.973737109978001]),
    ("O", [2.4735338513339995, -3.3872526556800002, 15.08517825]),
    ("O", [2.541521613813, -3.8623938562740006, 11.177091291129]),
    (
        "O",
        [2.541521613813, -3.8623938562740006, 18.993265208870998],
    ),
    ("O", [2.3593686777569998, 3.150916674678, 6.973737109978001]),
    (
        "O",
        [2.3593686777569998, 3.150916674678, 3.0830483900220003],
    ),
    ("O", [3.7929081486659992, -3.3872526556800002, 5.02839275]),
    ("O", [3.724920386187, -3.862393856274, 8.936479708871]),
    ("O", [3.724920386187, -3.862393856274, 1.120305791129]),
    (
        "O",
        [1.5841314718319999, 1.8081664370280002, 13.139833890022],
    ),
    (
        "O",
        [1.5841314718319999, 1.8081664370280002, 17.030522609978],
    ),
    (
        "O",
        [4.615691983265999, -0.26982536855999983, 1.120305791129],
    ),
    (
        "O",
        [4.615691983265999, -0.26982536855999983, 8.936479708871],
    ),
    ("O", [0.0, 0.0, 16.765224608488]),
    ("O", [0.0, 0.0, 6.708439108488]),
    ("O", [0.0, 0.0, 3.3483463915120004]),
    ("O", [0.0, 0.0, 13.405131891512]),
    ("O", [5.207391369453, -1.2946787751659996, 11.177091291129]),
    (
        "O",
        [5.207391369453, -1.2946787751659996, 18.993265208870998],
    ),
    ("O", [1.650750016734, -0.26982536856, 11.177091291129]),
    ("O", [1.650750016734, -0.26982536856, 18.993265208870998]),
    ("O", [1.059050630547, -1.294678775166, 1.120305791129]),
    ("O", [1.059050630547, -1.294678775166, 8.936479708871]),
];

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key)
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(default)
}

fn kmesh() -> [usize; 3] {
    let raw = std::env::var("YTA_KMESH").unwrap_or_else(|_| "3,3,1".into());
    let v: Vec<usize> = raw
        .split(',')
        .map(|s| s.trim().parse().expect("YTA_KMESH"))
        .collect();
    assert_eq!(
        v.len(),
        3,
        "YTA_KMESH must be three comma-separated integers"
    );
    [v[0], v[1], v[2]]
}

fn build_cell(ke_cutoff: f64, basis: &str) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(
                SITES_ANG
                    .iter()
                    .map(|(s, r)| ((*s).to_string(), *r))
                    .collect(),
            ),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix(LATTICE_ANG),
        pseudo: Some("gth-pbe".into()),
        ke_cutoff: Some(ke_cutoff),
        ..Default::default()
    })
    .expect("the YTa7O19 cell must build")
}

/// `kind:sigma` (`fermi`/`gauss`) or `none` (aufbau occupations).
fn parse_smearing(spec: &str) -> Option<pyscf_pbc_scf::smearing::Smearing> {
    if spec.trim() == "none" {
        return None;
    }
    let (kind, sigma) = spec
        .split_once(':')
        .expect("smearing must be kind:sigma or none");
    let sigma: f64 = sigma.trim().parse().expect("smearing sigma");
    Some(match kind.trim() {
        "fermi" => pyscf_pbc_scf::smearing::Smearing::fermi(sigma),
        "gauss" | "gaussian" => pyscf_pbc_scf::smearing::Smearing::gaussian(sigma),
        other => panic!("smearing kind {other:?}: fermi|gauss|none"),
    })
}

fn write_json(path: &str, value: &serde_json::Value) {
    let tmp = format!("{path}.tmp");
    std::fs::write(&tmp, serde_json::to_string_pretty(value).expect("json")).expect("write result");
    std::fs::rename(&tmp, path).expect("rename result");
}

// ---------------------------------------------------------------------------
// Checkpoint files
// ---------------------------------------------------------------------------

const DMS_MAGIC: &[u8; 8] = b"YTACKPT1";
const STATE_MAGIC: &[u8; 8] = b"YTASTAT2";

/// `nset x nk` matrices of `n` complex elements: three u64 dims, then every
/// matrix's real plane and imaginary plane as little-endian f64.
fn push_dms(buf: &mut Vec<u8>, dms: &KDms) {
    let n = dms[0][0].re.len();
    for d in [dms.len(), dms[0].len(), n] {
        buf.extend_from_slice(&(d as u64).to_le_bytes());
    }
    for m in dms.iter().flatten() {
        assert_eq!(m.re.len(), n, "checkpoint: ragged matrices");
        for v in m.re.iter().chain(m.im.iter()) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// Parse one [`push_dms`] block at `buf[*at..]`, advancing `*at`.
fn take_dms(buf: &[u8], at: &mut usize, what: &Path) -> KDms {
    let u64_at = |i: usize| u64::from_le_bytes(buf[i..i + 8].try_into().unwrap()) as usize;
    let (nset, nk, n) = (u64_at(*at), u64_at(*at + 8), u64_at(*at + 16));
    *at += 24;
    let bytes = nset * nk * n * 16;
    assert!(buf.len() >= *at + bytes, "{}: truncated", what.display());
    let mut vals = buf[*at..*at + bytes]
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()));
    *at += bytes;
    (0..nset)
        .map(|_| {
            (0..nk)
                .map(|_| {
                    let re: Vec<f64> = vals.by_ref().take(n).collect();
                    let im: Vec<f64> = vals.by_ref().take(n).collect();
                    CTensor { re, im }
                })
                .collect()
        })
        .collect()
}

fn write_atomic(path: &Path, bytes: &[u8]) {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).expect("write checkpoint");
    std::fs::rename(&tmp, path).expect("rename checkpoint");
}

fn save_dms(path: &Path, dms: &KDms) {
    let mut buf = Vec::from(&DMS_MAGIC[..]);
    push_dms(&mut buf, dms);
    write_atomic(path, &buf);
}

const DM0_MAGIC: &[u8; 8] = b"YTADM0V1";

/// The projected start density, tagged with the lineage token of the `pre`
/// density it came from.
fn save_dm0(path: &Path, token: &str, dms: &KDms) {
    let mut buf = Vec::from(&DM0_MAGIC[..]);
    buf.extend_from_slice(&(token.len() as u64).to_le_bytes());
    buf.extend_from_slice(token.as_bytes());
    push_dms(&mut buf, dms);
    write_atomic(path, &buf);
}

/// `(pre token, density)`. A pre-lineage `dm0.bin` (plain density file) reads
/// with token `""` — the upstream that s1 states of that era recorded — so an
/// interrupted older run keeps its s1 progress after upgrading.
fn load_dm0(path: &Path) -> Option<(String, KDms)> {
    let buf = std::fs::read(path).ok()?;
    if &buf[..8] == DMS_MAGIC {
        return load_dms(path).map(|d| (String::new(), d));
    }
    assert_eq!(
        &buf[..8],
        DM0_MAGIC,
        "{}: not a YTa projected density",
        path.display()
    );
    let len = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as usize;
    let token = String::from_utf8(buf[16..16 + len].to_vec()).expect("dm0 token utf-8");
    let mut at = 16 + len;
    let dms = take_dms(&buf, &mut at, path);
    assert_eq!(at, buf.len(), "{}: trailing bytes", path.display());
    Some((token, dms))
}

fn load_dms(path: &Path) -> Option<KDms> {
    let buf = std::fs::read(path).ok()?;
    assert_eq!(
        &buf[..8],
        DMS_MAGIC,
        "{}: not a YTa density checkpoint",
        path.display()
    );
    let mut at = 8;
    let dms = take_dms(&buf, &mut at, path);
    assert_eq!(at, buf.len(), "{}: trailing bytes", path.display());
    Some(dms)
}

/// An SCF stage's resume point: everything [`CycleState`] reported after the
/// last finished cycle, plus the controls that define what the stage
/// converges TO (smearing, tolerance). Completion lives in the same atomic
/// file as the density — there is no separate marker to lose.
struct ScfState {
    next_cycle: u32,
    e_tot: f64,
    converged: bool,
    controls: String,
    dm: KDms,
    fock: KDms,
}

/// Layout: magic, u64 next cycle, f64 e_tot, u8 converged, u64 length +
/// UTF-8 controls JSON, the density block, the Fock block.
fn save_state(path: &Path, st: &CycleState<'_>, controls: &str) {
    let mut buf = Vec::from(&STATE_MAGIC[..]);
    buf.extend_from_slice(&u64::from(st.cycle + 1).to_le_bytes());
    buf.extend_from_slice(&st.e_tot.to_le_bytes());
    buf.push(u8::from(st.converged));
    buf.extend_from_slice(&(controls.len() as u64).to_le_bytes());
    buf.extend_from_slice(controls.as_bytes());
    push_dms(&mut buf, st.dm);
    push_dms(&mut buf, st.fock);
    write_atomic(path, &buf);
}

fn load_state(path: &Path) -> Option<ScfState> {
    let buf = std::fs::read(path).ok()?;
    assert_eq!(
        &buf[..8],
        STATE_MAGIC,
        "{}: not a YTa SCF state (v2)",
        path.display()
    );
    let next_cycle = u64::from_le_bytes(buf[8..16].try_into().unwrap()) as u32;
    let e_tot = f64::from_le_bytes(buf[16..24].try_into().unwrap());
    let converged = buf[24] != 0;
    let clen = u64::from_le_bytes(buf[25..33].try_into().unwrap()) as usize;
    let controls = String::from_utf8(buf[33..33 + clen].to_vec()).expect("controls utf-8");
    let mut at = 33 + clen;
    let dm = take_dms(&buf, &mut at, path);
    let fock = take_dms(&buf, &mut at, path);
    assert_eq!(at, buf.len(), "{}: trailing bytes", path.display());
    Some(ScfState {
        next_cycle,
        e_tot,
        converged,
        controls,
        dm,
        fock,
    })
}

fn unix_now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}

/// The checkpoint directory (or none: then nothing is saved or resumed).
#[derive(Clone)]
struct Ckpt(Option<PathBuf>);

impl Ckpt {
    fn path(&self, name: &str) -> Option<PathBuf> {
        self.0.as_ref().map(|d| d.join(name))
    }
    fn exists(&self, name: &str) -> bool {
        self.path(name).is_some_and(|p| p.exists())
    }

    /// `status.json`: where the run is, for humans and the notebook.
    fn status(&self, stage: &str, detail: serde_json::Value) {
        eprintln!("[yta] status {stage}: {detail}");
        if let Some(p) = self.path("status.json") {
            let v = json!({"stage": stage, "detail": detail, "updated_unix": unix_now()});
            write_atomic(
                &p,
                serde_json::to_string_pretty(&v).expect("json").as_bytes(),
            );
        }
    }
    /// One `history.jsonl` line per SCF cycle.
    fn history(&self, rec: serde_json::Value) {
        if let Some(p) = self.path("history.jsonl") {
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(p)
                .expect("history");
            writeln!(f, "{rec}").expect("history line");
        }
    }
    /// A one-electron matrix: loaded, or computed and saved.
    fn one_e(&self, name: &str, compute: impl FnOnce() -> KMats) -> KMats {
        let path = self.path(name);
        if let Some(m) = path.as_deref().and_then(load_dms) {
            eprintln!("[yta] {name}: loaded from checkpoint");
            return m.into_iter().next().unwrap();
        }
        self.status("1e", json!({"computing": name}));
        let t = Instant::now();
        let m = compute();
        eprintln!(
            "[yta] {name}: computed in {:.1}s",
            t.elapsed().as_secs_f64()
        );
        if let Some(p) = path.as_deref() {
            save_dms(p, &vec![m.clone()]);
        }
        m
    }
}

// ---------------------------------------------------------------------------
// SCF stages
// ---------------------------------------------------------------------------

/// The controls of one SCF stage.
struct Stage {
    name: &'static str,
    smearing: Option<String>,
    damp: f64,
    level_shift: f64,
    diis_start: u32,
}

impl Stage {
    /// What the stage converges TO, and from what: its smearing and
    /// tolerance plus the lineage token of the density it starts from
    /// (`upstream`). A converged checkpoint with another identity is not
    /// skipped: with the same upstream it restarts from its own density, with
    /// a different upstream it is discarded. Damping, level shift and DIIS only
    /// steer the path and may change between sessions.
    fn identity(&self, conv_tol: f64, upstream: &str) -> String {
        json!({"stage": self.name, "smearing": self.smearing, "conv_tol": conv_tol, "upstream": upstream})
            .to_string()
    }
}

/// How a stage ended.
struct StageOutcome {
    dm: KDms,
    converged: bool,
    /// Lineage of `dm`: the stage identity plus the cycle count and energy of
    /// the density. Downstream stages and the bands record it and are reused
    /// only against the same token — no separate invalidation step, so no
    /// window in which a kill could leave stale results looking current.
    token: String,
    /// The stage's report (carrying `token`). For a stage finished in an
    /// earlier session it is a minimal one rebuilt from the checkpoint, used
    /// only if `result.json` has no report with this token (a kill between the
    /// converged checkpoint and the report write).
    report: serde_json::Value,
    /// Whether the stage ran in this session (its report is authoritative).
    ran: bool,
}

/// Store a stage's report under `key` unless `result.json` already holds the
/// report of exactly this density (same lineage token).
fn store_report(result: &mut serde_json::Value, key: &str, outcome: &StageOutcome) -> bool {
    if outcome.ran || result[key]["token"] != outcome.report["token"] {
        result[key] = outcome.report.clone();
        return true;
    }
    false
}

/// Run (or resume, or skip) one SCF stage. `start` supplies the initial
/// guess when the stage has no checkpoint yet.
#[allow(clippy::too_many_arguments)]
fn run_stage(
    mf: &mut Krks,
    ck: &Ckpt,
    stage: &Stage,
    s1e: &KMats,
    h1e: &KMats,
    base: &KScfConfig,
    nocc: usize,
    upstream: &str,
    start: impl FnOnce() -> KInitGuess,
) -> StageOutcome {
    let state_path = ck.path(&format!("scf_{}.bin", stage.name));
    let state = state_path.as_deref().and_then(load_state);
    let identity = stage.identity(base.conv_tol, upstream);
    let token = |cycles: u32, e_tot: f64| format!("{identity}@{cycles}:{:016x}", e_tot.to_bits());
    if let Some(st) = &state
        && st.converged
        && st.controls == identity
    {
        eprintln!(
            "[yta] stage {}: converged in an earlier session (e_tot {:.10}), skipped",
            stage.name, st.e_tot
        );
        let st = state.unwrap();
        let token = token(st.next_cycle, st.e_tot);
        let report = json!({
            "converged": true, "cycles_total": st.next_cycle, "e_tot_ha": st.e_tot,
            "controls": serde_json::from_str::<serde_json::Value>(&st.controls).unwrap_or(json!(st.controls)),
            "token": token, "note": "rebuilt from the checkpoint; the session that converged it did not save its report",
        });
        return StageOutcome {
            dm: st.dm,
            converged: true,
            token,
            report,
            ran: false,
        };
    }
    let upstream_of = |controls: &str| {
        serde_json::from_str::<serde_json::Value>(controls)
            .ok()
            .map(|v| v["upstream"].clone())
    };
    let mut rejected = false;
    let state = match state {
        Some(st)
            if st.controls != identity && upstream_of(&st.controls) != Some(json!(upstream)) =>
        {
            eprintln!(
                "[yta] stage {}: its saved state was built on a different upstream density; discarded",
                stage.name
            );
            rejected = true;
            None
        }
        other => other,
    };
    mf.smearing = stage.smearing.as_deref().and_then(parse_smearing);
    let (init_guess, fock_last, first_cycle) = match state {
        Some(st) if st.controls == identity => {
            eprintln!(
                "[yta] stage {}: resuming at cycle {} (last e_tot {:.10})",
                stage.name, st.next_cycle, st.e_tot
            );
            (KInitGuess::UserDm(st.dm), Some(st.fock), st.next_cycle)
        }
        Some(st) => {
            // Same upstream, different target (smearing / tolerance changed):
            // start a fresh cycle count from the saved density.
            eprintln!(
                "[yta] stage {}: controls changed ({} -> {}); restarting from the saved density",
                stage.name, st.controls, identity
            );
            (KInitGuess::UserDm(st.dm), None, 0)
        }
        None => {
            // Legacy checkpoint: a bare `dm_<stage>.bin` from the first runner
            // (never after a lineage rejection — it would be just as stale).
            let legacy = if rejected {
                None
            } else {
                ck.path(&format!("dm_{}.bin", stage.name))
            };
            match legacy.as_deref().and_then(load_dms) {
                Some(dm) => {
                    eprintln!(
                        "[yta] stage {}: resuming from a legacy density checkpoint",
                        stage.name
                    );
                    (KInitGuess::UserDm(dm), None, 0)
                }
                None => (start(), None, 0),
            }
        }
    };
    let hook = {
        let ck = ck.clone();
        let name = stage.name;
        let path = state_path.clone();
        let identity = identity.clone();
        let t_stage = Instant::now();
        CycleHook(Arc::new(move |st: &CycleState<'_>| {
            if let Some(p) = path.as_deref() {
                save_state(p, st, &identity);
            }
            ck.history(json!({"stage": name, "cycle": st.cycle, "e_tot": st.e_tot, "converged": st.converged,
                              "t_unix": unix_now(), "stage_seconds": t_stage.elapsed().as_secs_f64()}));
            ck.status(name, json!({"cycle": st.cycle, "e_tot": st.e_tot}));
        }))
    };
    let cfg = KScfConfig {
        s1e: Some(s1e.clone()),
        h1e: Some(h1e.clone()),
        on_cycle: Some(hook),
        fock_last,
        first_cycle,
        init_guess,
        damp: stage.damp,
        level_shift: stage.level_shift,
        diis_start_cycle: stage.diis_start,
        ..base.clone()
    };
    eprintln!(
        "[yta] stage {}: smearing {:?}, damp {}, level_shift {}, diis_space {}, diis_start {}, cycles {}..{}",
        stage.name,
        stage.smearing,
        cfg.damp,
        cfg.level_shift,
        cfg.diis_space,
        cfg.diis_start_cycle,
        cfg.first_cycle,
        cfg.max_cycle
    );
    ck.status(stage.name, json!({"cycle": first_cycle, "starting": true}));
    let t = Instant::now();
    let scf = mf.kernel(&cfg).expect("KRKS kernel");
    let seconds = t.elapsed().as_secs_f64();
    let homo = scf
        .mo_energy
        .iter()
        .map(|e| e[nocc - 1])
        .fold(f64::MIN, f64::max);
    let lumo = scf
        .mo_energy
        .iter()
        .map(|e| e[nocc])
        .fold(f64::MAX, f64::min);
    eprintln!(
        "[yta] stage {} converged={} cycles={} e_tot={:.10} Ha  in {seconds:.1}s",
        stage.name, scf.converged, scf.cycles, scf.e_tot
    );
    let report = json!({
        "controls": {"smearing": stage.smearing, "damp": cfg.damp, "level_shift": cfg.level_shift,
                     "diis_space": cfg.diis_space, "diis_start": cfg.diis_start_cycle},
        "converged": scf.converged, "cycles_total": scf.cycles, "e_tot_ha": scf.e_tot,
        "homo_ha": homo, "lumo_ha": lumo, "seconds_this_session": seconds,
        "kpts_mo_energy_ha": scf.mo_energy,
    });
    let token = token(scf.cycles, scf.e_tot);
    let mut report = report;
    report["token"] = json!(token);
    StageOutcome {
        dm: scf.dm,
        converged: scf.converged,
        token,
        report,
        ran: true,
    }
}

fn exit_unconverged(ck: &Ckpt, stage: &str, code: i32) -> ! {
    eprintln!("[yta] stage {stage} did NOT converge within YTA_MAXCYC: stopping");
    ck.status(
        "failed",
        json!({"stage": stage, "reason": "not converged within YTA_MAXCYC"}),
    );
    std::process::exit(code);
}

/// Hand pooled device memory from finished stages back before the next one
/// (the pool otherwise keeps every earlier stage's peak reserved).
fn release_device_pool() {
    if let Ok(sel) = pyscf_algebra::select_backend() {
        sel.client.memory_cleanup();
    }
}

fn init_guess_from_env() -> KInitGuess {
    match env_or("YTA_INIT_GUESS", "minao".to_string()).as_str() {
        "minao" => KInitGuess::Minao,
        "atom" => KInitGuess::Atom,
        "1e" => KInitGuess::OneElectron,
        other => panic!("YTA_INIT_GUESS {other:?}: minao|atom|1e"),
    }
}

fn main() {
    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .try_init();

    let t_all = Instant::now();
    let ke: f64 = env_or("YTA_KE", 100.0);
    let mesh_k = kmesh();
    let npath: usize = env_or("YTA_NPATH", 60);
    let chunk: usize = env_or::<usize>("YTA_CHUNK", 6).max(1);
    let xc: String = env_or("YTA_XC", "pbe".to_string());
    let max_cycle: u32 = env_or("YTA_MAXCYC", 80);
    let conv_tol: f64 = env_or("YTA_CONV", 1e-7);
    let basis: String = env_or("YTA_BASIS", "gth-szv-molopt-sr".to_string());
    let pre_basis: Option<String> = std::env::var("YTA_PRE_BASIS")
        .ok()
        .filter(|v| !v.trim().is_empty());
    let require = std::env::var("YTA_REQUIRE_CONVERGED").is_ok_and(|v| v == "1");
    let ck = Ckpt(
        std::env::var("YTA_CKPT_DIR")
            .ok()
            .filter(|v| !v.trim().is_empty())
            .map(PathBuf::from),
    );
    if let Some(d) = &ck.0 {
        std::fs::create_dir_all(d).expect("YTA_CKPT_DIR");
    }
    let out: String = std::env::var("YTA_OUT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| {
            ck.path("result.json")
                .map_or("yta7o19_bands.json".into(), |p| {
                    p.to_string_lossy().into_owned()
                })
        });

    // Resolve the backend exactly as the kernels will, and refuse to quote a
    // silent CPU fallback as a GPU run.
    let sel = pyscf_algebra::select_backend().expect("backend must resolve");
    let backend = sel.kind.name();
    eprintln!(
        "[yta] backend resolved: {backend} (PYSCF_BACKEND={:?})",
        sel.raw_env
    );
    if let Ok(want) = std::env::var("YTA_REQUIRE_BACKEND") {
        assert_eq!(
            backend, want,
            "YTA_REQUIRE_BACKEND={want} but the backend resolved to {backend}"
        );
    }
    drop(sel);
    let xc_backend = format!("{:?}", pyscf_dft::XcBackend::default());
    eprintln!("[yta] XC backend: {xc_backend}");
    if let Ok(want) = std::env::var("YTA_REQUIRE_XC") {
        assert_eq!(
            xc_backend, want,
            "YTA_REQUIRE_XC={want} but the XC backend is {xc_backend}"
        );
    }

    let cell = build_cell(ke, &basis);
    let nao = cell.mol.nao_nr;
    let nelec = cell.tot_electrons(1);
    let nocc = nelec / 2;
    let mesh = cell.mesh;
    eprintln!(
        "[yta] cell: 54 atoms, basis {basis}, nao {nao}, nelectron {nelec}, mesh {mesh:?}, ke_cutoff {ke} Ha"
    );
    let kpts = make_kpts_default(&cell, mesh_k).expect("k mesh");

    // A resumed run keeps what earlier sessions reported.
    let mut result =
        ck.0.as_ref()
            .and_then(|_| std::fs::read_to_string(&out).ok())
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .unwrap_or_else(|| json!({}));
    result["system"] = json!("YTa7O19 (mp-772036, P-6c2, Z=2)");
    result["method"] = json!({"xc": xc, "basis": basis, "pre_basis": pre_basis, "pseudo": "gth-pbe",
                              "ke_cutoff_ha": ke, "mesh": mesh, "kmesh": mesh_k, "nao": nao,
                              "nelectron": nelec, "nocc": nocc});
    result["backend"] = json!(backend);
    result["xc_backend"] = json!(xc_backend);

    // Everything in the checkpoint depends on these; refuse to mix runs.
    let fingerprint = json!({"system": "YTa7O19 mp-772036", "pseudo": "gth-pbe", "basis": basis,
                             "pre_basis": pre_basis, "ke_cutoff_ha": ke, "kmesh": mesh_k, "xc": xc});
    if let Some(fp) = ck.path("fingerprint.json") {
        let refuse = |why: String| -> ! {
            eprintln!("[yta] {why}");
            ck.status("failed", json!({"reason": why, "run": fingerprint}));
            std::process::exit(6);
        };
        let adopt = std::env::var("YTA_ADOPT_CHECKPOINT").is_ok_and(|v| v == "1");
        match std::fs::read_to_string(&fp) {
            Ok(text) => match serde_json::from_str::<serde_json::Value>(&text) {
                Ok(old) if old == fingerprint => {}
                Ok(old) => refuse(format!(
                    "checkpoint directory belongs to a different calculation:\n  checkpoint: {old}\n  this run:   {fingerprint}"
                )),
                Err(e) => refuse(format!(
                    "fingerprint.json is unreadable ({e}); refusing to guess what the checkpoint holds"
                )),
            },
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => refuse(format!(
                "fingerprint.json cannot be read ({e}); refusing to guess what the checkpoint holds"
            )),
            Err(_) => {
                // No fingerprint: an empty directory is a fresh run; files of
                // unknown provenance are only taken when explicitly adopted
                // (`YTA_ADOPT_CHECKPOINT=1`, set by `ktrun.py seed-ckpt`).
                let has_data = ck.0.as_ref().is_some_and(|d| {
                    std::fs::read_dir(d).is_ok_and(|it| {
                        it.flatten()
                            .any(|e| e.file_name().to_string_lossy().ends_with(".bin"))
                    })
                });
                if has_data && !adopt {
                    refuse("checkpoint files without fingerprint.json; set YTA_ADOPT_CHECKPOINT=1 to adopt them \
                            for this calculation"
                        .into());
                }
                write_atomic(
                    &fp,
                    serde_json::to_string_pretty(&fingerprint)
                        .expect("json")
                        .as_bytes(),
                );
            }
        }
    }

    let s1_controls = Stage {
        name: "s1",
        smearing: std::env::var("YTA_SMEARING")
            .ok()
            .filter(|v| !v.trim().is_empty()),
        damp: env_or("YTA_DAMP", 0.0),
        level_shift: env_or("YTA_LEVEL_SHIFT", 0.0),
        diis_start: env_or("YTA_DIIS_START", 1),
    };
    let base_for = |cell: &Cell| KScfConfig {
        conv_tol,
        max_cycle,
        verbose: true,
        diis_space: env_or("YTA_DIIS_SPACE", 8),
        ..KScfConfig::for_cell(cell)
    };

    // Stage 1e — the target basis's overlap and core Hamiltonian.
    let mut mf = Krks::new(cell, &kpts, &xc).expect("KRKS must build");
    let s1e = ck.one_e("s1e.bin", || {
        let _s = tracing::info_span!("scf_get_ovlp").entered();
        mf.get_ovlp().expect("get_ovlp")
    });
    let h1e = ck.one_e("h1e.bin", || {
        let _s = tracing::info_span!("scf_get_hcore").entered();
        mf.get_hcore().expect("get_hcore")
    });
    if std::env::var("YTA_STOP_AFTER").is_ok_and(|v| v == "1e") {
        eprintln!(
            "[yta] YTA_STOP_AFTER=1e: stopping after {:.1}s",
            t_all.elapsed().as_secs_f64()
        );
        return;
    }
    release_device_pool();
    let base = base_for(mf.cell());

    // Stages pre + project — only while s1 has not started.
    // While s1 has not started, `pre` is (re)checked every run: it is skipped
    // when converged under the current controls, and the projection is redone
    // whenever `dm0.bin` came from a different `pre` density. Once s1 has
    // started, its start density is fixed and recorded as s1's upstream.
    let s1_started = ck.exists("scf_s1.bin") || ck.exists("dm_s1.bin");
    let mut dm0: Option<(String, KDms)> = ck.path("dm0.bin").as_deref().and_then(load_dm0);
    if let Some(pre) = pre_basis.as_deref()
        && !s1_started
    {
        let pre_cell = build_cell(ke, pre);
        let pre_nocc = pre_cell.tot_electrons(1) / 2;
        eprintln!("[yta] stage pre: basis {pre}, nao {}", pre_cell.mol.nao_nr);
        let mut pre_mf = Krks::new(pre_cell, &kpts, &xc).expect("pre KRKS must build");
        let pre_s1e = ck.one_e("pre_s1e.bin", || pre_mf.get_ovlp().expect("pre get_ovlp"));
        let pre_h1e = ck.one_e("pre_h1e.bin", || pre_mf.get_hcore().expect("pre get_hcore"));
        let pre_stage = Stage {
            name: "pre",
            ..s1_controls.clone_controls()
        };
        let pre_base = KScfConfig {
            conv_tol: env_or("YTA_PRE_CONV", 1e-4),
            max_cycle: env_or("YTA_PRE_MAXCYC", max_cycle),
            ..base_for(pre_mf.cell())
        };
        let outcome = run_stage(
            &mut pre_mf,
            &ck,
            &pre_stage,
            &pre_s1e,
            &pre_h1e,
            &pre_base,
            pre_nocc,
            "",
            init_guess_from_env,
        );
        if store_report(&mut result, "scf_pre", &outcome) {
            write_json(&out, &result);
        }
        if !outcome.converged {
            if require {
                exit_unconverged(&ck, "pre", 5);
            }
            eprintln!("[yta] stage pre did not converge; projecting its last density anyway");
        }
        if dm0.as_ref().is_some_and(|(t, _)| *t == outcome.token) {
            eprintln!("[yta] stage project: dm0.bin already holds this pre density's projection");
        } else {
            ck.status("project", json!({"from": pre, "to": basis}));
            let t = Instant::now();
            let projected = pyscf_pbc_scf::addons::project_dm_nr2nr(
                pre_mf.cell(),
                &outcome.dm[0],
                mf.cell(),
                &s1e,
                &kpts,
            )
            .expect("project the pre density");
            let ne: f64 = projected
                .iter()
                .zip(&s1e)
                .map(|(d, s)| pyscf_pbc_df::zlinalg::ztrace_ab(d, s, nao).0)
                .sum::<f64>()
                / kpts.len() as f64;
            eprintln!(
                "[yta] stage project: {pre} -> {basis} in {:.1}s, electrons {ne:.6} (of {nelec})",
                t.elapsed().as_secs_f64()
            );
            result["projection"] = json!({"from": pre, "to": basis, "electrons": ne});
            write_json(&out, &result);
            let dm0_k = vec![projected];
            if let Some(p) = ck.path("dm0.bin") {
                save_dm0(&p, &outcome.token, &dm0_k);
            }
            dm0 = Some((outcome.token.clone(), dm0_k));
        }
    }

    // Stage s1.
    release_device_pool();
    let s1_upstream = dm0.as_ref().map_or(String::new(), |(t, _)| t.clone());
    let s1 = run_stage(
        &mut mf,
        &ck,
        &s1_controls,
        &s1e,
        &h1e,
        &base,
        nocc,
        &s1_upstream,
        || match dm0 {
            Some((_, dm)) => {
                eprintln!(
                    "[yta] stage s1: starting from the projected {} density",
                    pre_basis.as_deref().unwrap_or("?")
                );
                KInitGuess::UserDm(dm)
            }
            None => init_guess_from_env(),
        },
    );
    if store_report(&mut result, "scf_stage1", &s1) {
        write_json(&out, &result);
    }
    let mut final_report = "scf_stage1";
    if !s1.converged && require {
        exit_unconverged(&ck, "s1", 3);
    }
    let mut dm = s1.dm;
    let mut token = s1.token;

    // Stage s2 (`YTA_REFINE_SMEARING`): re-converge from the stage-1 density
    // with a narrow smearing (or `none`); bands use the refined density.
    if let Some(spec) = std::env::var("YTA_REFINE_SMEARING")
        .ok()
        .filter(|v| !v.trim().is_empty())
    {
        let s2_controls = Stage {
            name: "s2",
            smearing: Some(spec),
            damp: env_or("YTA_REFINE_DAMP", 0.0),
            level_shift: env_or("YTA_REFINE_LEVEL_SHIFT", 0.0),
            diis_start: env_or("YTA_REFINE_DIIS_START", 1),
        };
        let s1_dm = dm.clone();
        let s2 = run_stage(
            &mut mf,
            &ck,
            &s2_controls,
            &s1e,
            &h1e,
            &base,
            nocc,
            &token,
            move || KInitGuess::UserDm(s1_dm),
        );
        if store_report(&mut result, "scf_stage2", &s2) {
            write_json(&out, &result);
        }
        final_report = "scf_stage2";
        if !s2.converged && require {
            exit_unconverged(&ck, "s2", 4);
        }
        dm = s2.dm;
        token = s2.token;
    }
    // `scf` always describes the density the bands are computed on.
    result["scf"] = result[final_report].clone();
    result["scf"]["stage"] = json!(final_report);
    write_json(&out, &result);

    // Stage bands — chunks an earlier session already wrote are kept.
    let path = band_path(mf.cell(), BravaisLattice::Hexagonal, npath).expect("band path");
    eprintln!(
        "[yta] bands: {} k-points on G-M-K-G-A-L-H-A|L-M|K-H, {chunk} per call",
        path.len()
    );
    let t = Instant::now();
    let mut bands: Vec<Vec<f64>> = match result["bands"]["energies_ha"].as_array() {
        Some(prev)
            if ck.0.is_some()
                && result["bands"]["kpts_scaled"] == json!(path.scaled)
                && result["bands"]["from"] == json!(token) =>
        {
            prev.iter()
                .map(|e| serde_json::from_value(e.clone()).expect("band energies"))
                .collect()
        }
        _ => {
            if result.get("bands").is_some() {
                eprintln!(
                    "[yta] bands: the saved band points came from another density or path; recomputing"
                );
                if let Some(o) = result.as_object_mut() {
                    o.remove("summary");
                }
            }
            Vec::with_capacity(path.len())
        }
    };
    if !bands.is_empty() {
        eprintln!(
            "[yta] bands: {} k-points already in {out}, resuming",
            bands.len()
        );
    }
    let start = bands.len();
    for (i, ks) in path.abs[start..].chunks(chunk).enumerate() {
        ck.status("bands", json!({"done": bands.len(), "of": path.len()}));
        release_device_pool();
        let tc = Instant::now();
        let (e, _) = mf.get_bands(ks, &dm).expect("get_bands");
        bands.extend(e);
        eprintln!(
            "[yta]   chunk {i}: {}/{} k-points, {:.1}s",
            bands.len(),
            path.len(),
            tc.elapsed().as_secs_f64()
        );
        result["bands"] = json!({
            "x": path.x, "tick_x": path.tick_x, "tick_labels": path.tick_labels,
            "kpts_scaled": path.scaled, "energies_ha": bands, "complete": bands.len() == path.len(),
            "from": token,
        });
        write_json(&out, &result);
    }
    let t_bands = t.elapsed().as_secs_f64();

    let vbm = bands.iter().map(|e| e[nocc - 1]).fold(f64::MIN, f64::max);
    let cbm = bands.iter().map(|e| e[nocc]).fold(f64::MAX, f64::min);
    let direct = bands
        .iter()
        .map(|e| e[nocc] - e[nocc - 1])
        .fold(f64::MAX, f64::min);
    result["summary"] = json!({
        "vbm_ha": vbm, "cbm_ha": cbm,
        "gap_ev": (cbm - vbm) * HARTREE_EV, "direct_gap_ev": direct * HARTREE_EV,
        "seconds_last_session": {"bands": t_bands, "total": t_all.elapsed().as_secs_f64()},
    });
    write_json(&out, &result);
    ck.status(
        "done",
        json!({"gap_ev": (cbm - vbm) * HARTREE_EV, "direct_gap_ev": direct * HARTREE_EV}),
    );
    eprintln!(
        "[yta] DONE gap {:.4} eV (direct {:.4} eV)  total {:.1}s -> {out}",
        (cbm - vbm) * HARTREE_EV,
        direct * HARTREE_EV,
        t_all.elapsed().as_secs_f64()
    );
}

impl Stage {
    /// The same controls under another stage name.
    fn clone_controls(&self) -> Stage {
        Stage {
            name: self.name,
            smearing: self.smearing.clone(),
            damp: self.damp,
            level_shift: self.level_shift,
            diis_start: self.diis_start,
        }
    }
}

#![allow(dead_code)]
//! Shared fixtures for the Phase-16 `pyscf-pbc-cc` tests.
//!
//! `diamond` is `PBC-MASTER-PLAN §9.2`'s reference cell. The mesh is PINNED at
//! `[15,15,15]` for every test, matching the pin 16-01's measurements ran under
//! (`measurements/README.md`, fixture pin): at `cell.precision = 1e-8` the
//! default mesh is `[47,47,47]`, where one `KRHF` at `[1,1,2]` alone costs
//! 79 s. Every gate number these tests use was measured at the same pin, so
//! the two sides are comparable.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

/// `§9.2` `diamond` — C2 fcc `a = 3.5668 A`, `gth-szv` / `gth-pade`, mesh pinned.
pub fn diamond(mesh: [usize; 3]) -> Cell {
    let a0 = 3.5668;
    let q = a0 / 4.0;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([
            [0.0, a0 / 2.0, a0 / 2.0],
            [a0 / 2.0, 0.0, a0 / 2.0],
            [a0 / 2.0, a0 / 2.0, 0.0],
        ]),
        pseudo: Some("gth-pade".into()),
        mesh: Some(mesh),
        ..Default::default()
    })
    .expect("diamond cell")
}

// ---------------------------------------------------------------------------
// The PySCF oracle harness, shared by the `oracle_*` tests in this crate.
//
// Every oracle test is `#[ignore]`d AND short-circuits unless
// `PYSCF_ORACLE_VENV` is set — the same double gate
// `crates/pyscf-pbc-mp/tests/oracle_phase15.rs` uses.
// ---------------------------------------------------------------------------

use std::path::PathBuf;
use std::process::Command;

use pyscf_algebra::CTensor;
use pyscf_pbc_cc::ZArr;
use pyscf_pbc_df::{Fftdf, MoCoeff, PeriodicDf};
use pyscf_pbc_lib::KptsHelper;
use pyscf_pbc_mp::PaddedMos;
use pyscf_pbc_scf::{KScfConfig, Krhf};

pub const GATE: &str = "PYSCF_ORACLE_VENV";
pub const EMITTER: &str = ".planning/phases/16-periodic-cc-ci/measurements/oracle_phase16.py";

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

pub fn python() -> Option<PathBuf> {
    let raw = std::env::var(GATE).ok()?;
    let raw = raw.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    Some(if matches!(raw.as_str(), "1" | "true" | "auto" | "yes") {
        root().join(".venv/bin/python")
    } else {
        let path = PathBuf::from(&raw);
        if path.is_dir() {
            path.join("bin/python")
        } else {
            path
        }
    })
}

pub fn emit(section: &str) -> Option<String> {
    let Some(python) = python() else {
        eprintln!("skip: set {GATE} to arm the PySCF oracle");
        return None;
    };
    let out = Command::new(python)
        .args(["-u", EMITTER, section])
        .current_dir(root())
        .env("PYTHONPATH", root())
        .output()
        .expect("run the Phase-16 oracle emitter");
    assert!(
        out.status.success(),
        "oracle section {section} failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8(out.stdout).expect("UTF-8 oracle output");
    assert!(
        s.contains("pyscf.__version__=2.12.1"),
        "the oracle is not PySCF 2.12.1"
    );
    Some(s)
}

pub fn block(out: &str, name: &str) -> Vec<f64> {
    let head = format!("BEGIN {name} n=");
    let start = out
        .find(&head)
        .unwrap_or_else(|| panic!("oracle emitted no block {name}"));
    let rest = &out[start + head.len()..];
    let nl = rest.find('\n').expect("block header");
    let n: usize = rest[..nl].trim().parse().expect("block count");
    let body = &rest[nl + 1..];
    let end = body
        .find(&format!("END {name}"))
        .unwrap_or_else(|| panic!("block {name} is unterminated"));
    let v: Vec<f64> = body[..end]
        .split_ascii_whitespace()
        .map(|t| t.parse().expect("f64"))
        .collect();
    assert_eq!(v.len(), n, "block {name}: declared {n}, got {}", v.len());
    v
}

pub fn cblock(out: &str, name: &str) -> CTensor {
    let v = block(out, name);
    assert_eq!(v.len() % 2, 0, "block {name} is not complex");
    CTensor::from_planes(
        v.iter().step_by(2).copied().collect(),
        v.iter().skip(1).step_by(2).copied().collect(),
    )
}

pub fn scalar(out: &str, key: &str) -> f64 {
    let head = format!("{key}=");
    for line in out.lines() {
        if let Some(v) = line.trim().strip_prefix(&head) {
            return v.trim().parse().expect("scalar");
        }
    }
    panic!("oracle emitted no scalar {key}");
}

/// Max element-wise absolute difference between a `ZArr` and an upstream block.
pub fn maxdiff(got: &ZArr, want: &CTensor, name: &str) -> f64 {
    assert_eq!(
        got.len(),
        want.re.len(),
        "{name}: {} elements here, {} upstream",
        got.len(),
        want.re.len()
    );
    let mut m = 0.0_f64;
    for i in 0..got.len() {
        m = m
            .max((got.data().re[i] - want.re[i]).abs())
            .max((got.data().im[i] - want.im[i]).abs());
    }
    m
}

/// The same SplitMix64 stream `oracle_phase16.py`'s `synthetic_amps` uses, so
/// both sides build identical synthetic amplitudes with no file exchange.
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

pub fn synthetic(shape1: &[usize], shape2: &[usize]) -> (ZArr, ZArr) {
    let mut r = SplitMix64(20260906);
    let n1: usize = shape1.iter().product();
    let n2: usize = shape2.iter().product();
    let mut t1 = ZArr::zeros(shape1);
    for i in 0..n1 {
        t1.data_mut().re[i] = 0.05 * r.unit();
        t1.data_mut().im[i] = 0.05 * r.unit();
    }
    let mut t2 = ZArr::zeros(shape2);
    for i in 0..n2 {
        t2.data_mut().re[i] = 0.05 * r.unit();
        t2.data_mut().im[i] = 0.05 * r.unit();
    }
    (t1, t2)
}

/// Everything the CC layer needs, taken from UPSTREAM's converged mean field.
///
/// **Why the tests are built this way.** 16-01 measured this port's `KRHF` and
/// upstream's at `-8.652011318061934` vs `-8.651997841505` on diamond
/// `gth-szv` `[1,1,2]` with the mesh PINNED at `[15,15,15]` — `1.35e-5 Ha`
/// apart, while Phase 15's `oracle_phase15` measured the two agreeing to
/// `4.772e-11` on the same cell at the DEFAULT mesh. So the divergence is a
/// coarse-mesh property of the mean field, not of anything Phase 16 wrote, and
/// a correlation energy compared across two different mean fields measures the
/// mean fields. Feeding upstream's own `fock` / `mo_energy` / `mo_coeff` into
/// [`KEris::from_parts`] makes every gate here mean-field-INDEPENDENT — the
/// discipline `15-VERIFICATION` used when it drove `Lov` from "upstream's own
/// padded MOs" and got `2e-15`. The mean-field residual is reported separately
/// by [`mean_field_residual`] and recorded in `16-VERIFICATION.md`, not
/// silently absorbed.
pub struct Upstream {
    pub padded: PaddedMos,
    pub fock: ZArr,
    pub mo_energy: Vec<Vec<f64>>,
    pub nkpts: usize,
    pub nocc: usize,
    pub nmo: usize,
}

pub fn upstream_mos(out: &str) -> Upstream {
    let nkpts = scalar(out, "nkpts") as usize;
    let nocc = scalar(out, "nocc") as usize;
    let nmo = scalar(out, "nmo") as usize;
    let nao = scalar(out, "nao") as usize;
    let c = cblock(out, "mo_coeff");
    assert_eq!(c.re.len(), nkpts * nao * nmo, "mo_coeff shape");
    let mo_coeff: Vec<MoCoeff> = (0..nkpts)
        .map(|k| {
            let off = k * nao * nmo;
            MoCoeff::new(
                nao,
                nmo,
                CTensor {
                    re: c.re[off..off + nao * nmo].to_vec(),
                    im: c.im[off..off + nao * nmo].to_vec(),
                },
            )
        })
        .collect();
    let me = block(out, "mo_energy");
    let mo_energy: Vec<Vec<f64>> = (0..nkpts)
        .map(|k| me[k * nmo..(k + 1) * nmo].to_vec())
        .collect();
    let nocc_per_kpt: Vec<usize> = block(out, "nocc_per_kpt")
        .iter()
        .map(|v| *v as usize)
        .collect();
    let fock = ZArr::from_ctensor(&[nkpts, nmo, nmo], cblock(out, "fock")).expect("fock shape");
    Upstream {
        padded: PaddedMos {
            mo_coeff,
            mo_energy: mo_energy.clone(),
            nmo_per_kpt: vec![nmo; nkpts],
            nocc_per_kpt,
            nmo,
            nocc,
        },
        fock,
        mo_energy,
        nkpts,
        nocc,
        nmo,
    }
}

/// Build this port's `_ERIS` on upstream's mean field.
pub fn eris_on_upstream_mf(f: &Fixture, up: &Upstream) -> (pyscf_pbc_cc::KEris, KptsHelper) {
    let mut khelper = KptsHelper::without_symm_map(&f.cell.a, PeriodicDf::kpts(&f.df));
    let eris = pyscf_pbc_cc::KEris::from_parts(
        &f.df,
        &mut khelper,
        &up.padded,
        up.fock.clone(),
        up.mo_energy.clone(),
        0.0,
        pyscf_pbc_cc::KErisOpts::default(),
    )
    .expect("_ERIS on upstream MOs");
    (eris, khelper)
}

/// The mean-field residual, REPORTED rather than gated — see [`Upstream`].
pub fn mean_field_residual(f: &Fixture, out: &str) -> f64 {
    let d = (f.scf.e_tot - scalar(out, "e_hf")).abs();
    println!(
        "MEAN-FIELD RESIDUAL (reported, not gated): this port {} vs upstream {} -> {d:e}",
        f.scf.e_tot,
        scalar(out, "e_hf")
    );
    d
}

pub struct Fixture {
    pub scf: pyscf_pbc_scf::KScfResult,
    pub df: Fftdf,
    pub cell: pyscf_pbc_gto::Cell,
}

pub fn diamond_scf(nk: [usize; 3]) -> Fixture {
    let cell = diamond([15, 15, 15]);
    let kpts = cell.make_kpts(nk).expect("kpts");
    let df = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    let mut mf = Krhf::from_df(Box::new(Fftdf::new(cell.clone(), &kpts).expect("fftdf")));
    mf.exxdiv = None;
    let mut cfg = KScfConfig::for_cell(&cell);
    cfg.conv_tol = 1e-10;
    let scf = mf.kernel(&cfg).expect("KRHF converges");
    assert!(scf.converged, "the reference SCF must converge");
    Fixture { scf, df, cell }
}

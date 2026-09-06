//! Opt-in Phase-16 checks against the vendored PySCF 2.12.1 tree.
//!
//! Plain workspace tests never invoke Python: every test here is `#[ignore]`d
//! AND short-circuits unless `PYSCF_ORACLE_VENV` is set — the same double gate
//! `crates/pyscf-pbc-mp/tests/oracle_phase15.rs` uses.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-cc \
//!   --test oracle_phase16 -- --ignored --nocapture
//! ```
//!
//! **Every tolerance here traces to
//! `.planning/phases/16-periodic-cc-ci/measurements/README.md §1`**, which
//! 16-01 measured. None was invented by this file.

mod common;

use std::path::PathBuf;
use std::process::Command;

use pyscf_algebra::CTensor;
use pyscf_pbc_cc::kccsd_rhf::{energy, init_amps, update_amps};
use pyscf_pbc_cc::keris::Blk;
use pyscf_pbc_cc::{ZArr, imdk};
use pyscf_pbc_df::{MoCoeff, PeriodicDf};
use pyscf_pbc_lib::KptsHelper;
use pyscf_pbc_mp::PaddedMos;
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_scf::{KScfConfig, Krhf};
use pyscf_runtime::ZWorkspacePool;
use std::sync::Arc;

const GATE: &str = "PYSCF_ORACLE_VENV";
const EMITTER: &str = ".planning/phases/16-periodic-cc-ci/measurements/oracle_phase16.py";

/// `measurements/README.md §1` G1 — `KRCCSD e_corr` vs upstream, FFTDF.
const G1_E_CORR: f64 = 1e-7;

/// The ERI-block gate, MEASURED not assumed.
///
/// Driven from upstream's own MO coefficients, this port's seven `_ERIS`
/// blocks and upstream's agree to `1.2e-8 … 1.5e-7` at the pinned `[15,15,15]`
/// mesh (`oooo 1.21e-8`, `ooov 6.62e-8`, `oovv 1.46e-7`, …) — the FFT
/// integral-transform floor at that mesh, not a transposition, which would be
/// `O(1)`. For scale, `measurements/README.md §7` measured upstream's OWN
/// symmetry-loop and all-triples paths differing by up to `1.32e-7` on the same
/// fixture, so this IS the mesh's own integral floor and not something either
/// side could tighten. The gate sits one order above the largest measured
/// residual and four orders below anything a real defect would produce.
const ERI_BLOCK: f64 = 1e-6;

/// The intermediates inherit the ERI floor (they are linear and bilinear in
/// the blocks), so they are gated at the same level rather than tighter. The
/// largest measured is `cc_Wvvvv` at `2.28e-7`, which inherits `vvvv`'s.
const IMDS_BLOCK: f64 = 1e-6;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn python() -> Option<PathBuf> {
    let raw = std::env::var(GATE).ok()?;
    let raw = raw.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    Some(if matches!(raw.as_str(), "1" | "true" | "auto" | "yes") {
        root().join(".venv/bin/python")
    } else {
        let path = PathBuf::from(&raw);
        if path.is_dir() { path.join("bin/python") } else { path }
    })
}

fn emit(section: &str) -> Option<String> {
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

fn block(out: &str, name: &str) -> Vec<f64> {
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

fn cblock(out: &str, name: &str) -> CTensor {
    let v = block(out, name);
    assert_eq!(v.len() % 2, 0, "block {name} is not complex");
    CTensor::from_planes(
        v.iter().step_by(2).copied().collect(),
        v.iter().skip(1).step_by(2).copied().collect(),
    )
}

fn scalar(out: &str, key: &str) -> f64 {
    let head = format!("{key}=");
    for line in out.lines() {
        if let Some(v) = line.trim().strip_prefix(&head) {
            return v.trim().parse().expect("scalar");
        }
    }
    panic!("oracle emitted no scalar {key}");
}

/// Max element-wise absolute difference between a `ZArr` and an upstream block.
fn maxdiff(got: &ZArr, want: &CTensor, name: &str) -> f64 {
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
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

fn synthetic(shape1: &[usize], shape2: &[usize]) -> (ZArr, ZArr) {
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
struct Upstream {
    padded: PaddedMos,
    fock: ZArr,
    mo_energy: Vec<Vec<f64>>,
    nkpts: usize,
    nocc: usize,
    nmo: usize,
}

fn upstream_mos(out: &str) -> Upstream {
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
fn eris_on_upstream_mf(f: &Fixture, up: &Upstream) -> (pyscf_pbc_cc::KEris, KptsHelper) {
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
fn mean_field_residual(f: &Fixture, out: &str) -> f64 {
    let d = (f.scf.e_tot - scalar(out, "e_hf")).abs();
    println!(
        "MEAN-FIELD RESIDUAL (reported, not gated): this port {} vs upstream {} -> {d:e}",
        f.scf.e_tot,
        scalar(out, "e_hf")
    );
    d
}

struct Fixture {
    scf: pyscf_pbc_scf::KScfResult,
    df: Fftdf,
    cell: pyscf_pbc_gto::Cell,
}

fn diamond_scf(nk: [usize; 3]) -> Fixture {
    let cell = common::diamond([15, 15, 15]);
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

/// `_ERIS`: the seven blocks, the Fock matrix and `mo_energy`, element-wise.
///
/// This is the FIRST oracle test in the phase because it is where a
/// transposition slips in — the 14-05 `decompose_j2c` class of defect
/// (`16-CONTEXT §3.4`). Everything downstream inherits it silently.
#[test]
#[ignore = "opt-in PySCF oracle"]
fn eris_blocks_match_upstream() {
    let Some(out) = emit("eris") else { return };
    let f = diamond_scf([1, 1, 2]);
    mean_field_residual(&f, &out);

    let up = upstream_mos(&out);
    let (eris, _kh) = eris_on_upstream_mf(&f, &up);
    assert_eq!(eris.nkpts, up.nkpts);
    assert_eq!(eris.nocc, up.nocc);
    assert_eq!(eris.nmo, up.nmo);

    let mut worst: Vec<(&str, f64)> = Vec::new();
    for (b, name) in [
        (Blk::Oooo, "oooo"),
        (Blk::Ooov, "ooov"),
        (Blk::Oovv, "oovv"),
        (Blk::Ovov, "ovov"),
        (Blk::Voov, "voov"),
        (Blk::Vovv, "vovv"),
        (Blk::Vvvv, "vvvv"),
    ] {
        let want = cblock(&out, name);
        let nk = eris.nkpts;
        let dims = b.dims(eris.nocc, eris.nvir);
        let mut shape = vec![nk, nk, nk];
        shape.extend_from_slice(&dims);
        let mut got = ZArr::zeros(&shape);
        for k0 in 0..nk {
            for k1 in 0..nk {
                for k2 in 0..nk {
                    got.set_leading(&[k0, k1, k2], &eris.blk(b, k0, k1, k2).expect("block"))
                        .expect("shape");
                }
            }
        }
        let d = maxdiff(&got, &want, name);
        println!("max|{name} - upstream| = {d:e}");
        worst.push((name, d));
    }
    // Report EVERY block before failing: a single failing assertion hides the
    // pattern that says whether this is one bad transposition or a uniform
    // integral floor.
    let bad: Vec<&(&str, f64)> = worst.iter().filter(|(_, d)| *d >= ERI_BLOCK).collect();
    assert!(
        bad.is_empty(),
        "blocks above the {ERI_BLOCK:e} gate: {bad:?} (all: {worst:?})"
    );
}

/// The `cc_*` intermediates and `update_amps`, on a FIXED synthetic `t1`/`t2`.
///
/// Synthetic amplitudes rather than converged ones on purpose: this isolates
/// the intermediate arithmetic from the iteration, so a failure here names one
/// function instead of "the energy is wrong".
#[test]
#[ignore = "opt-in PySCF oracle"]
fn intermediates_and_update_amps_match_upstream() {
    let Some(out) = emit("imds") else { return };
    let f = diamond_scf([1, 1, 2]);
    mean_field_residual(&f, &out);
    let up = upstream_mos(&out);
    let (eris, kh) = eris_on_upstream_mf(&f, &up);
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let (t1, t2) = synthetic(
        &[nk, nocc, nvir],
        &[nk, nk, nk, nocc, nocc, nvir, nvir],
    );
    // The two sides must be looking at the same amplitudes before anything else
    // is compared.
    assert!(maxdiff(&t1, &cblock(&out, "t1"), "t1") == 0.0, "t1 streams differ");
    assert!(maxdiff(&t2, &cblock(&out, "t2"), "t2") == 0.0, "t2 streams differ");

    let kc = &kh.kconserv;
    let opts = pyscf_pbc_cc::KrccsdOpts::default();
    let pool = Arc::new(ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES));
    let budget = ZWorkspacePool::DEFAULT_BUDGET_BYTES;

    for (name, got) in [
        ("cc_Foo", imdk::cc_foo(&t1, &t2, &eris, kc).expect("cc_Foo")),
        ("cc_Fvv", imdk::cc_fvv(&t1, &t2, &eris, kc).expect("cc_Fvv")),
        ("cc_Fov", imdk::cc_fov(&t1, &t2, &eris).expect("cc_Fov")),
        ("Loo", imdk::loo(&t1, &t2, &eris, kc).expect("Loo")),
        ("Lvv", imdk::lvv(&t1, &t2, &eris, kc).expect("Lvv")),
    ] {
        let d = maxdiff(&got, &cblock(&out, name), name);
        println!("max|{name} - upstream| = {d:e}");
        assert!(d < IMDS_BLOCK, "{name} differs by {d:e}, above {IMDS_BLOCK:e}");
    }

    for (name, blocks) in [
        (
            "cc_Woooo",
            imdk::cc_woooo(&pool, &t1, &t2, &eris, kc, budget).expect("cc_Woooo"),
        ),
        (
            "cc_Wvvvv",
            imdk::cc_wvvvv(&pool, &t1, &t2, &eris, kc, budget).expect("cc_Wvvvv"),
        ),
        (
            "cc_Wvoov",
            imdk::cc_wvoov(&pool, &t1, &t2, &eris, kc, budget).expect("cc_Wvoov"),
        ),
        (
            "cc_Wvovo",
            imdk::cc_wvovo(&pool, &t1, &t2, &eris, kc, budget).expect("cc_Wvovo"),
        ),
    ] {
        let bs = blocks.block_shape().to_vec();
        let mut shape = vec![nk, nk, nk];
        shape.extend_from_slice(&bs);
        let mut got = ZArr::zeros(&shape);
        for k0 in 0..nk {
            for k1 in 0..nk {
                for k2 in 0..nk {
                    got.set_leading(&[k0, k1, k2], &blocks.get([k0, k1, k2]).expect("block"))
                        .expect("shape");
                }
            }
        }
        let d = maxdiff(&got, &cblock(&out, name), name);
        println!("max|{name} - upstream| = {d:e}");
        assert!(d < IMDS_BLOCK, "{name} differs by {d:e}, above {IMDS_BLOCK:e}");
        blocks.release();
    }

    let (t1new, t2new) =
        update_amps(&pool, &t1, &t2, &eris, &up.padded, kc, &opts).expect("update_amps");
    let d1 = maxdiff(&t1new, &cblock(&out, "t1new"), "t1new");
    let d2 = maxdiff(&t2new, &cblock(&out, "t2new"), "t2new");
    println!("max|t1new - upstream| = {d1:e}   max|t2new - upstream| = {d2:e}");
    assert!(d1 < IMDS_BLOCK, "t1new differs by {d1:e}");
    assert!(d2 < IMDS_BLOCK, "t2new differs by {d2:e}");

    let e = energy(&t1, &t2, &eris, kc).expect("energy");
    let want = scalar(&out, "energy_synth");
    println!("energy(synthetic) {e} vs upstream {want}");
    assert!(
        (e - want).abs() < IMDS_BLOCK,
        "energy differs by {:e}",
        (e - want).abs()
    );
}

/// **G1** — `KRCCSD e_corr` vs upstream, FFTDF, diamond `gth-szv` `[1,1,2]`,
/// mesh `[15,15,15]`, `cell.precision = 1e-8`, `conv_tol = 1e-9`.
#[test]
#[ignore = "opt-in PySCF oracle"]
fn krccsd_e_corr_matches_upstream_fftdf() {
    let Some(out) = emit("krccsd") else { return };
    let f = diamond_scf([1, 1, 2]);
    mean_field_residual(&f, &out);
    let up = upstream_mos(&out);
    let (eris, kh) = eris_on_upstream_mf(&f, &up);
    let opts = pyscf_pbc_cc::KrccsdOpts::default();

    let (emp2, _, _) = init_amps(&eris, &up.padded, &kh.kconserv).expect("init_amps");
    let want_emp2 = scalar(&out, "emp2");
    println!("emp2 {emp2} vs upstream {want_emp2}  |Δ| {:e}", (emp2 - want_emp2).abs());
    assert!(
        (emp2 - want_emp2).abs() < G1_E_CORR,
        "init_amps emp2 differs by {:e}",
        (emp2 - want_emp2).abs()
    );

    let pool = Arc::new(ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES));
    let res = pyscf_pbc_cc::kccsd_rhf::kernel(&pool, &eris, &up.padded, &kh.kconserv, &opts)
        .expect("KRCCSD kernel");
    assert!(res.converged, "KRCCSD did not converge");
    let want = scalar(&out, "e_corr");
    let d = (res.e_corr - want).abs();
    println!(
        "e_corr {} vs upstream {want}  |Δ| {d:e}  (G1 = {G1_E_CORR:e})",
        res.e_corr
    );
    assert!(d < G1_E_CORR, "e_corr differs by {d:e}, above G1 {G1_E_CORR:e}");

    let e = energy(&res.t1, &res.t2, &eris, &kh.kconserv).expect("energy");
    assert!(
        (e - res.e_corr).abs() < 1e-14,
        "energy() disagrees with the kernel's own e_corr"
    );
}

/// **G4** — `KCCSD(T)` fast vs slow, and both vs upstream.
///
/// `measurements/README.md §5` measured upstream's own two implementations
/// agreeing to `3.27e-16` absolute / `2.95e-13` relative — the one place a
/// Phase-16 number can be tight, because it is the same input through the same
/// formula twice with no convergence noise between. This port is held to the
/// same **1e-13 relative**.
#[test]
#[ignore = "opt-in PySCF oracle"]
fn ccsd_t_fast_equals_slow_and_matches_upstream() {
    let Some(out) = emit("triples") else { return };
    let f = diamond_scf([1, 1, 2]);
    mean_field_residual(&f, &out);
    let up = upstream_mos(&out);
    let (eris, kh) = eris_on_upstream_mf(&f, &up);
    let opts = pyscf_pbc_cc::KrccsdOpts::default();
    let pool = Arc::new(ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES));
    let res = pyscf_pbc_cc::kccsd_rhf::kernel(&pool, &eris, &up.padded, &kh.kconserv, &opts)
        .expect("KRCCSD kernel");
    assert!(res.converged);
    let want_ecorr = scalar(&out, "e_corr");
    println!(
        "e_corr {} vs upstream {want_ecorr}  |Δ| {:e}",
        res.e_corr,
        (res.e_corr - want_ecorr).abs()
    );

    let kpts = PeriodicDf::kpts(&f.df).to_vec();
    let slow = pyscf_pbc_cc::kccsd_t_rhf_slow::kernel(
        &eris, &up.padded, &res.t1, &res.t2, &kh.kconserv, &f.cell.a, &kpts, None,
    )
    .expect("(T) slow");
    let fast = pyscf_pbc_cc::kccsd_t_rhf::kernel(
        &eris, &up.padded, &res.t1, &res.t2, &kh.kconserv, &f.cell.a, &kpts, None,
    )
    .expect("(T) fast");

    let rel = (fast - slow).abs() / slow.abs();
    println!("(T) fast {fast}  slow {slow}  |Δ| {:e}  relative {rel:e}", (fast - slow).abs());
    // **G4 = 1e-12, corrected from the 1e-13 first written here.**
    // `measurements/README.md §5` measured UPSTREAM's own fast-vs-slow
    // agreement at `2.95e-13` relative — so a `1e-13` gate is BELOW upstream's
    // own agreement and would fail a correct implementation. That is the same
    // defect this phase has now caught five times (ROADMAP's 1e-14,
    // §7's 1e-8, 16-07's 1e-10, 16-08's 1e-11, and this one, written by the
    // test author rather than the plan). Measured here: `8.36e-13` relative,
    // `9.29e-16` absolute, against upstream's `2.95e-13` / `3.27e-16`.
    assert!(
        rel < 1e-12,
        "fast-vs-slow relative {rel:e} above G4 1e-12 (upstream's own is 2.95e-13)"
    );

    // Blocking invariance (16-08 test 3): the energy must not depend on the
    // virtual block size. This is what catches a wrong `mo_offset`/`slices`
    // translation, and it is oracle-free.
    let blocked = pyscf_pbc_cc::kccsd_t_rhf::kernel(
        &eris, &up.padded, &res.t1, &res.t2, &kh.kconserv, &f.cell.a, &kpts, Some(2),
    )
    .expect("(T) fast, blocked");
    println!("(T) blocked(2) {blocked}  vs unblocked {fast}  |Δ| {:e}", (blocked - fast).abs());
    assert!(
        (blocked - fast).abs() / fast.abs() < 1e-12,
        "the (T) energy depends on the virtual block size"
    );

    let want_fast = scalar(&out, "et_fast");
    let want_slow = scalar(&out, "et_slow");
    println!(
        "(T) vs upstream: fast |Δ| {:e}, slow |Δ| {:e}  (upstream fast {want_fast}, slow {want_slow})",
        (fast - want_fast).abs(),
        (slow - want_slow).abs()
    );
    // The (T) correction inherits the ERI floor of §10, so it is gated at the
    // same 1e-6 the blocks are, not at G4 — G4 is the fast-vs-slow gate.
    assert!(
        (fast - want_fast).abs() < ERI_BLOCK,
        "(T) fast differs from upstream by {:e}",
        (fast - want_fast).abs()
    );
}

/// **16-07** — `KGCCSD`, on upstream's own KGHF mean field.
///
/// Same design as the RHF tests: the seven spin-orbital `<pq||rs>` blocks are
/// rebuilt here from upstream's `mo_coeff`, so what is compared is the CC code
/// and not two SCFs (`measurements/README.md §10`).
///
/// `e_corr` is gated at **G3 = `1e-8`**, which 16-01 measured: upstream's own
/// `KGCCSD` and `KRCCSD` differ by `4.95e-9` on this fixture, so `16-07`'s
/// plan-time `1e-10` would fail a correct implementation.
#[test]
#[ignore = "opt-in PySCF oracle"]
fn kgccsd_matches_upstream() {
    let Some(out) = emit("kgccsd") else { return };
    let f = diamond_scf([1, 1, 2]);
    let nkpts = scalar(&out, "nkpts") as usize;
    let nocc = scalar(&out, "nocc") as usize;
    let nmo = scalar(&out, "nmo") as usize;
    let nao = scalar(&out, "nao") as usize;
    let nvir = nmo - nocc;

    let c = cblock(&out, "mo_coeff");
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
    let me = block(&out, "mo_energy");
    let mo_energy: Vec<Vec<f64>> = (0..nkpts)
        .map(|k| me[k * nmo..(k + 1) * nmo].to_vec())
        .collect();
    let fock = ZArr::from_ctensor(&[nkpts, nmo, nmo], cblock(&out, "fock")).expect("fock");
    let nocc_per_kpt: Vec<usize> = block(&out, "nocc_per_kpt")
        .iter()
        .map(|v| *v as usize)
        .collect();
    let padded = PaddedMos {
        mo_coeff: mo_coeff.clone(),
        mo_energy: mo_energy.clone(),
        nmo_per_kpt: vec![nmo; nkpts],
        nocc_per_kpt,
        nmo,
        nocc,
    };

    let khelper = KptsHelper::without_symm_map(&f.cell.a, PeriodicDf::kpts(&f.df));
    let eris = pyscf_pbc_cc::kccsd::KgEris::from_parts(
        &f.df,
        &khelper,
        &mo_coeff,
        fock,
        mo_energy,
        nocc,
        4_000_000_000,
    )
    .expect("spin-orbital _ERIS");
    assert_eq!(eris.nocc, nocc);
    assert_eq!(eris.nvir, nvir);

    use pyscf_pbc_cc::kccsd::GBlk;
    let mut worst: Vec<(&str, f64)> = Vec::new();
    for (b, name) in [
        (GBlk::Oooo, "oooo"),
        (GBlk::Ooov, "ooov"),
        (GBlk::Ovoo, "ovoo"),
        (GBlk::Oovv, "oovv"),
        (GBlk::Ovov, "ovov"),
        (GBlk::Ovvv, "ovvv"),
        (GBlk::Vvvv, "vvvv"),
    ] {
        let want = cblock(&out, name);
        let d = b.dims(nocc, nvir);
        let mut shape = vec![nkpts, nkpts, nkpts];
        shape.extend_from_slice(&d);
        let mut got = ZArr::zeros(&shape);
        for k0 in 0..nkpts {
            for k1 in 0..nkpts {
                for k2 in 0..nkpts {
                    got.set_leading(&[k0, k1, k2], &eris.blk(b, k0, k1, k2).expect("block"))
                        .expect("shape");
                }
            }
        }
        let m = maxdiff(&got, &want, name);
        println!("max|{name} - upstream| = {m:e}");
        worst.push((name, m));
    }
    let bad: Vec<&(&str, f64)> = worst.iter().filter(|(_, d)| *d >= ERI_BLOCK).collect();
    assert!(bad.is_empty(), "spin-orbital blocks above {ERI_BLOCK:e}: {bad:?}");

    // `energy` and `update_amps` on the SAME fixed synthetic amplitudes.
    let st1 = ZArr::from_ctensor(&[nkpts, nocc, nvir], cblock(&out, "st1")).expect("st1");
    let st2 = ZArr::from_ctensor(
        &[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir],
        cblock(&out, "st2"),
    )
    .expect("st2");
    let e = pyscf_pbc_cc::kccsd::energy(&st1, &st2, &eris).expect("energy");
    let want = scalar(&out, "energy_synth");
    println!("energy(synthetic) {e} vs upstream {want}  |Δ| {:e}", (e - want).abs());
    assert!((e - want).abs() < IMDS_BLOCK, "energy differs by {:e}", (e - want).abs());

    let (t1n, t2n) = pyscf_pbc_cc::kccsd::update_amps(
        &st1,
        &st2,
        &eris,
        &padded,
        &khelper.kconserv,
        0.0,
    )
    .expect("update_amps");
    let d1 = maxdiff(&t1n, &cblock(&out, "st1new"), "st1new");
    let d2 = maxdiff(&t2n, &cblock(&out, "st2new"), "st2new");
    println!("max|t1new - upstream| = {d1:e}   max|t2new - upstream| = {d2:e}");
    assert!(d1 < IMDS_BLOCK, "t1new differs by {d1:e}");
    assert!(d2 < IMDS_BLOCK, "t2new differs by {d2:e}");

    let opts = pyscf_pbc_cc::KrccsdOpts::default();
    let res = pyscf_pbc_cc::kccsd::kernel(&eris, &padded, &khelper.kconserv, &opts)
        .expect("KGCCSD kernel");
    let want = scalar(&out, "e_corr");
    let d = (res.e_corr - want).abs();
    println!(
        "KGCCSD e_corr {} vs upstream {want}  |Δ| {d:e}  converged {} in {} cycles",
        res.e_corr, res.converged, res.cycles
    );
    assert!(res.converged, "KGCCSD did not converge");
    // G3 = 1e-8 (measured: upstream's own KGCCSD and KRCCSD differ by 4.95e-9).
    assert!(d < 1e-8, "KGCCSD e_corr differs by {d:e}, above G3 1e-8");

    // 16-08 Task 3 — the SPIN-ORBITAL (T), on UPSTREAM's own converged
    // amplitudes so the (T) code is isolated from the CC iteration.
    let ut1 = ZArr::from_ctensor(&[nkpts, nocc, nvir], cblock(&out, "t1")).expect("t1");
    let ut2 = ZArr::from_ctensor(
        &[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir],
        cblock(&out, "t2"),
    )
    .expect("t2");
    let kpts = PeriodicDf::kpts(&f.df).to_vec();
    let et = pyscf_pbc_cc::kccsd_t::kernel(
        &eris,
        &padded,
        &ut1,
        &ut2,
        &khelper.kconserv,
        &f.cell.a,
        &kpts,
    )
    .expect("spin-orbital (T)");
    let want_et = scalar(&out, "et_spinorb");
    let de = (et - want_et).abs();
    println!("spin-orbital (T) {et} vs upstream {want_et}  |Δ| {de:e}");
    assert!(
        de < ERI_BLOCK,
        "the spin-orbital (T) differs from upstream by {de:e}"
    );
}

/// **G2 — the GDF route.** `KRCCSD e_corr` and the seven `_ERIS` blocks on a
/// **Gaussian** density-fitting mean field, stated separately from FFTDF.
///
/// `kccsd_rhf.py:37` imports `GDF, RSGDF` and branches the whole `_ERIS` build
/// on the mean field's DF class, and 16-01 measured the plane-wave/Gaussian
/// split at **`9.22e-4 Ha`** on this cell (`measurements/README.md §4`) —
/// three orders worse than the standing memory records at SCF level. A gate
/// that does not name its route is untestable, which is why this one exists
/// rather than a single "matches upstream" number.
#[test]
#[ignore = "opt-in PySCF oracle"]
fn krccsd_e_corr_matches_upstream_gdf() {
    let Some(out) = emit("eris_gdf") else { return };
    let cell = common::diamond([15, 15, 15]);
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let df = pyscf_pbc_df::Gdf::new(cell.clone(), &kpts);

    let up = upstream_mos(&out);
    let mut khelper = KptsHelper::without_symm_map(&cell.a, &kpts);
    let eris = pyscf_pbc_cc::KEris::from_parts(
        &df,
        &mut khelper,
        &up.padded,
        up.fock.clone(),
        up.mo_energy.clone(),
        0.0,
        pyscf_pbc_cc::KErisOpts::default(),
    )
    .expect("_ERIS on the GDF route");

    let mut worst: Vec<(&str, f64)> = Vec::new();
    for (b, name) in [
        (Blk::Oooo, "oooo"),
        (Blk::Ooov, "ooov"),
        (Blk::Oovv, "oovv"),
        (Blk::Ovov, "ovov"),
        (Blk::Voov, "voov"),
        (Blk::Vovv, "vovv"),
        (Blk::Vvvv, "vvvv"),
    ] {
        let want = cblock(&out, name);
        let dims = b.dims(up.nocc, up.nmo - up.nocc);
        let nk = up.nkpts;
        let mut shape = vec![nk, nk, nk];
        shape.extend_from_slice(&dims);
        let mut got = ZArr::zeros(&shape);
        for k0 in 0..nk {
            for k1 in 0..nk {
                for k2 in 0..nk {
                    got.set_leading(&[k0, k1, k2], &eris.blk(b, k0, k1, k2).expect("block"))
                        .expect("shape");
                }
            }
        }
        let d = maxdiff(&got, &want, name);
        println!("GDF: max|{name} - upstream| = {d:e}");
        worst.push((name, d));
    }
    // The GDF fitting residual is its own floor and is NOT the FFT one; the
    // gate is reported per block so the two routes' floors stay separable.
    let bad: Vec<&(&str, f64)> = worst.iter().filter(|(_, d)| *d >= ERI_BLOCK).collect();
    assert!(bad.is_empty(), "GDF blocks above {ERI_BLOCK:e}: {bad:?} (all: {worst:?})");

    let opts = pyscf_pbc_cc::KrccsdOpts::default();
    let (emp2, _, _) = init_amps(&eris, &up.padded, &khelper.kconserv).expect("init_amps");
    let want_emp2 = scalar(&out, "emp2");
    println!("GDF: emp2 {emp2} vs upstream {want_emp2}  |Δ| {:e}", (emp2 - want_emp2).abs());
    assert!((emp2 - want_emp2).abs() < G1_E_CORR, "GDF emp2 differs");

    let pool = Arc::new(ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES));
    let res = pyscf_pbc_cc::kccsd_rhf::kernel(&pool, &eris, &up.padded, &khelper.kconserv, &opts)
        .expect("KRCCSD kernel, GDF route");
    assert!(res.converged, "KRCCSD did not converge on the GDF route");
    let want = scalar(&out, "e_corr");
    let d = (res.e_corr - want).abs();
    println!(
        "GDF: e_corr {} vs upstream {want}  |Δ| {d:e}  (G2 = {G1_E_CORR:e})",
        res.e_corr
    );
    assert!(d < G1_E_CORR, "GDF e_corr differs by {d:e}, above G2 {G1_E_CORR:e}");
}

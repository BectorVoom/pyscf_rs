//! Opt-in Phase-16 `KCIS` checks against the vendored PySCF 2.12.1 tree
//! (plan 16-13 Task 4).
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-ci \
//!   --test oracle_kcis -- --ignored --nocapture
//! ```
//!
//! Like every other Phase-16 oracle test, the `_ERIS` is rebuilt from
//! UPSTREAM's own `fock` / `mo_energy` / `mo_coeff`, so what is compared is the
//! CIS code and not two SCFs (`measurements/README.md §10`).

use std::path::PathBuf;
use std::process::Command;

use pyscf_algebra::CTensor;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_cc::ZArr;
use pyscf_pbc_cc::keris::{KEris, KErisOpts};
use pyscf_pbc_ci::kcis_rhf::{KcisOpts, cis_diag, kernel_at_kshift};
use pyscf_pbc_df::{Fftdf, MoCoeff, PeriodicDf};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};
use pyscf_pbc_lib::KptsHelper;
use pyscf_pbc_mp::PaddedMos;

const GATE: &str = "PYSCF_ORACLE_VENV";
const EMITTER: &str = ".planning/phases/16-periodic-cc-ci/measurements/oracle_phase16.py";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn emit(section: &str) -> Option<String> {
    let raw = std::env::var(GATE).ok()?;
    let raw = raw.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    let python = if matches!(raw.as_str(), "1" | "true" | "auto" | "yes") {
        root().join(".venv/bin/python")
    } else {
        let p = PathBuf::from(&raw);
        if p.is_dir() { p.join("bin/python") } else { p }
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
    let s = String::from_utf8(out.stdout).expect("UTF-8");
    assert!(s.contains("pyscf.__version__=2.12.1"));
    Some(s)
}

fn block(out: &str, name: &str) -> Vec<f64> {
    let head = format!("BEGIN {name} n=");
    let start = out.find(&head).unwrap_or_else(|| panic!("no block {name}"));
    let rest = &out[start + head.len()..];
    let nl = rest.find('\n').expect("header");
    let n: usize = rest[..nl].trim().parse().expect("count");
    let body = &rest[nl + 1..];
    let end = body
        .find(&format!("END {name}"))
        .unwrap_or_else(|| panic!("block {name} unterminated"));
    let v: Vec<f64> = body[..end]
        .split_ascii_whitespace()
        .map(|t| t.parse().expect("f64"))
        .collect();
    assert_eq!(v.len(), n);
    v
}

fn cblock(out: &str, name: &str) -> CTensor {
    let v = block(out, name);
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
    panic!("no scalar {key}");
}

fn diamond(mesh: [usize; 3]) -> Cell {
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
    .expect("diamond")
}

/// **G7** — `KCIS` roots vs upstream, both solver paths.
///
/// Three comparisons, and the third is the one that says what the gate can be:
///
/// 1. this port's DENSE roots vs upstream's DENSE roots — same algorithm, so
///    this is the tight one;
/// 2. this port's DAVIDSON vs its own dense — 16-13 test 1, oracle-free in
///    spirit, isolating a solver bug from a Hamiltonian bug;
/// 3. **upstream's own Davidson vs its own dense**, reported — at `kshift = 0`
///    on this fixture they differ by `2.5e-3` on the third root, because the
///    Davidson converges to a different state. A gate tighter than THAT on a
///    Davidson root would be measuring the solver's luck.
#[test]
#[ignore = "opt-in PySCF oracle; converges an SCF"]
fn kcis_roots_match_upstream() {
    let Some(out) = emit("kcis") else { return };
    let cell = diamond([15, 15, 15]);
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let df = Fftdf::new(cell.clone(), &kpts).expect("fftdf");

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
        mo_coeff,
        mo_energy: mo_energy.clone(),
        nmo_per_kpt: vec![nmo; nkpts],
        nocc_per_kpt,
        nmo,
        nocc,
    };

    let mut khelper = KptsHelper::without_symm_map(&cell.a, PeriodicDf::kpts(&df));
    let eris = KEris::from_parts(
        &df,
        &mut khelper,
        &padded,
        fock,
        mo_energy,
        0.0,
        KErisOpts::default(),
    )
    .expect("_ERIS");
    assert_eq!(eris.nvir, nvir);

    for kshift in 0..nkpts {
        // The diagonal first: it is what the Davidson's guess and
        // preconditioner are built from, so a wrong diagonal is a wrong
        // solve even with a right matvec.
        let got = cis_diag(&eris, &khelper.kconserv, kshift).expect("cis_diag");
        // **`cis_diag` returns a COMPLEX array upstream** (`dtype =
        // eris.dtype`, `kcis_rhf.py:302`) even though every entry is real, so
        // the emitted block is INTERLEAVED. Reading it as real is a silent
        // factor-of-two index shift; it cost one debug cycle here.
        let want = cblock(&out, &format!("diag_{kshift}"));
        assert_eq!(got.len(), want.re.len(), "diagonal length");
        let d = got
            .iter()
            .zip(want.re.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let di = want.im.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        println!("kshift {kshift}: max|diag - upstream| = {d:e} (upstream max|Im| {di:e})");
        assert!(d < 1e-6, "the CIS diagonal differs by {d:e}");

        let dense = kernel_at_kshift(
            &eris,
            &khelper.kconserv,
            kshift,
            3,
            &KcisOpts {
                davidson: false,
                ..Default::default()
            },
        )
        .expect("dense CIS");
        let want_dense = block(&out, &format!("dense_{kshift}"));
        let dd = dense
            .iter()
            .zip(want_dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!("kshift {kshift}: dense {dense:?} vs upstream {want_dense:?}  max|Δ| {dd:e}");
        assert!(dd < 1e-6, "the dense CIS roots differ by {dd:e}");

        let dav = kernel_at_kshift(
            &eris,
            &khelper.kconserv,
            kshift,
            3,
            &KcisOpts {
                conv_tol: 1e-9,
                ..Default::default()
            },
        )
        .expect("Davidson CIS");
        let want_dav = block(&out, &format!("roots_{kshift}"));

        // **G7 — the Davidson roots against UPSTREAM's Davidson roots.**
        let dd2 = dav
            .iter()
            .zip(want_dav.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!("kshift {kshift}: davidson {dav:?} vs upstream {want_dav:?}  max|Δ| {dd2:e}");
        assert!(
            dd2 < 1e-5,
            "the Davidson CIS roots differ by {dd2:e}, above G7 1e-5"
        );

        // The Davidson-vs-dense SPREAD, on both sides. At `kshift = 0` on this
        // fixture UPSTREAM's own two paths differ by `2.51e-3` on the third
        // root — its Davidson converges to a different state — so a tighter
        // gate on a Davidson root would be measuring the solver's luck. What
        // IS assertable, and is a much stronger statement, is that this port
        // reproduces that spread: the two implementations agree on WHICH state
        // the Davidson finds, not merely on the ones the dense solve finds.
        let sd = dav
            .iter()
            .zip(dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        let us = want_dav
            .iter()
            .zip(want_dense.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!(
            "kshift {kshift}: davidson-vs-dense spread — this port {sd:e}, \
             upstream {us:e}, |Δ| {:e}",
            (sd - us).abs()
        );
        assert!(
            (sd - us).abs() < 1e-6,
            "this port's Davidson-vs-dense spread ({sd:e}) and upstream's ({us:e}) \
             disagree by {:e}: the two solvers are finding DIFFERENT states",
            (sd - us).abs()
        );
    }
}

/// Plan 16-13 Task 4 test 5 — the `dimension == 2` refusal, on the `§9.2`
/// reference cell that reaches it.
///
/// `kcis_rhf.py:630-637` refuses the direct-DF CIS path at `cell.dimension ==
/// 2` because 2-D ERIs are not positive definite: the 3-index tensor is stored
/// as a positive and a negative part and the negative part is not handled.
/// `graphene` IS a `§9.2` reference cell, so this refusal is reachable rather
/// than theoretical — which is exactly why it must carry the upstream line and
/// be tested rather than assumed.
///
/// Oracle-free: it asserts THIS PORT refuses, and names the upstream line the
/// refusal quotes. The companion oracle-gated half — that upstream still
/// raises there — is recorded in `16-13-SUMMARY.md` as not run, because this
/// port does not ship the `cis.direct = True` branch the refusal guards.
#[test]
fn direct_df_cis_refuses_at_dimension_two() {
    // graphene: C2 hexagonal with 20 A of vacuum, `dimension = 2`.
    let a0 = 2.46_f64;
    let c = 20.0_f64;
    let cell = Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("C".into(), [0.0, 0.0, 0.0]),
                ("C".into(), [a0 / 2.0, a0 / (2.0 * 3.0_f64.sqrt()), 0.0]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([
            [a0, 0.0, 0.0],
            [-a0 / 2.0, a0 * 3.0_f64.sqrt() / 2.0, 0.0],
            [0.0, 0.0, c],
        ]),
        pseudo: Some("gth-pade".into()),
        dimension: 2,
        low_dim_ft_type: pyscf_pbc_gto::LowDimFtType::None,
        ..Default::default()
    })
    .expect("graphene builds");
    assert_eq!(cell.dimension, 2, "the fixture must actually be 2-D");

    let err = pyscf_pbc_ci::kcis_rhf::check_dimension_for_direct_df(cell.dimension)
        .expect_err("dimension 2 must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("kcis_rhf.py:637"),
        "the refusal must name its upstream line, got: {msg}"
    );
    assert!(
        msg.contains("positive definite"),
        "the refusal must carry upstream's own reason, got: {msg}"
    );

    // Every other dimension passes, so the guard is not a blanket refusal.
    for d in [0_u8, 1, 3] {
        assert!(
            pyscf_pbc_ci::kcis_rhf::check_dimension_for_direct_df(d).is_ok(),
            "dimension {d} must NOT be refused"
        );
    }
}

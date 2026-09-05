//! Opt-in Phase 15 rollup checks against the vendored PySCF 2.12.1 tree.
//!
//! `15-07-PLAN.md` Task 1's nine parts:
//!
//! | # | what | test |
//! |---|---|---|
//! | 1 | `KptsHelper.symm_map` + `_operation` | [`symm_map_and_operation`] |
//! | 2 | `padding_k_idx`/`get_nocc`/`get_nmo`/`get_frozen_mask` | [`padding_surface`] |
//! | 3 | `ao2mo` + `ao2mo_7d`, one quadruple per implementor | [`ao2mo_and_ao2mo_7d`] |
//! | 4 | `Lov`, element-wise | [`lov_blocks`] |
//! | 5 | `e_corr`/`ss`/`os`, both routes, both systems | [`kmp2_energies`] |
//! | 6 | `t2`, `make_rdm1` both kinds, `gamma1_intermediates` | [`t2_rdm1_and_gamma1`] |
//! | 7 | `KMP2_stagger e_corr` | [`stagger_energies`] |
//! | 8 | the KUMP2 refusal | [`upstream_kump2_kernel_remains_an_explicit_refusal`] |
//! | 9 | the MO-first `ao2mo` block | [`mo_first_ao2mo_block`] |
//!
//! Plain workspace tests never invoke Python: every test is ignored and also
//! short-circuits unless `PYSCF_ORACLE_VENV` is set.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-mp \
//!   --test oracle_phase15 -- --ignored --nocapture
//! ```

mod common;

use std::path::PathBuf;
use std::process::Command;

use pyscf_algebra::CTensor;
use pyscf_mp2::Frozen;
use pyscf_pbc_df::{Aftdf, Fftdf, Gdf, Mdf, MoCoeff, PeriodicDf};
use pyscf_pbc_lib::KptsHelper;
use pyscf_pbc_mp::{
    FrozenK, Kmp2, PaddingIdx, PaddingKind, RdmKind, build_lov, gamma1_intermediates,
    get_frozen_mask, get_nmo, get_nocc, padding_k_idx,
};
use pyscf_pbc_scf::{KScfConfig, Krhf};

const GATE: &str = "PYSCF_ORACLE_VENV";
const ROLLUP: &str = ".planning/phases/15-periodic-ao2mo-kmp2/measurements/oracle_rollup.py";

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn python() -> Option<PathBuf> {
    let raw = std::env::var(GATE).ok()?;
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    Some(if matches!(raw, "1" | "true" | "auto" | "yes") {
        root().join(".venv/bin/python")
    } else {
        let path = PathBuf::from(raw);
        if path.is_dir() {
            path.join("bin/python")
        } else {
            path
        }
    })
}

fn run(args: &[&str]) -> Option<String> {
    let Some(python) = python() else {
        eprintln!("skip: set {GATE} to arm the PySCF oracle");
        return None;
    };
    let output = Command::new(python)
        .args(args)
        .current_dir(root())
        .env("PYTHONPATH", root())
        .output()
        .expect("run oracle Python");
    assert!(
        output.status.success(),
        "oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 oracle output");
    assert!(
        stdout.contains("pyscf.__version__=2.12.1"),
        "the oracle is not PySCF 2.12.1"
    );
    Some(stdout)
}

fn rollup(section: &str) -> Option<String> {
    run(&["-u", ROLLUP, section])
}

/// Extract one `BEGIN <name> n=<count> ... END <name>` block.
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

/// The same block, re-interleaved into a complex tensor.
fn cblock(out: &str, name: &str) -> CTensor {
    let v = block(out, name);
    assert_eq!(v.len() % 2, 0, "block {name} is not complex");
    CTensor::from_planes(
        v.iter().step_by(2).copied().collect(),
        v.iter().skip(1).step_by(2).copied().collect(),
    )
}

fn scalar<T: std::str::FromStr>(out: &str, key: &str) -> T
where
    T::Err: std::fmt::Debug,
{
    let head = format!("{key}=");
    for line in out.lines() {
        for tok in line.split_ascii_whitespace() {
            if let Some(v) = tok.strip_prefix(&head) {
                return v.parse().expect("scalar");
            }
        }
    }
    panic!("oracle emitted no scalar {key}");
}

/// A scalar on the line that starts with `prefix` — the KMP2 section prints
/// several `ss=`/`os=` pairs, one per route, and a bare key would always find
/// the first.
fn scalar_on<T: std::str::FromStr>(out: &str, prefix: &str, key: &str) -> T
where
    T::Err: std::fmt::Debug,
{
    let head = format!("{key}=");
    for line in out.lines().filter(|l| l.starts_with(prefix)) {
        for tok in line.split_ascii_whitespace() {
            if let Some(v) = tok.strip_prefix(&head) {
                return v.parse().expect("scalar");
            }
        }
    }
    panic!("oracle emitted no {key} on a line starting {prefix}");
}

fn max_dev(a: &CTensor, b: &CTensor) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch in a max_dev");
    a.re.iter()
        .zip(&b.re)
        .map(|(x, y)| (x - y).abs())
        .chain(a.im.iter().zip(&b.im).map(|(x, y)| (x - y).abs()))
        .fold(0.0, f64::max)
}

fn mo_from(out: &str, name: &str, nao: usize) -> MoCoeff {
    let c = cblock(out, name);
    let nmo = c.len() / nao;
    MoCoeff::new(nao, nmo, c)
}

// ---------------------------------------------------------------- 1. symm_map

#[test]
#[ignore = "requires PYSCF_ORACLE_VENV"]
fn symm_map_and_operation() {
    let Some(out) = rollup("symm_map") else {
        return;
    };
    let cell = common::diamond_anchor();
    for (tag, mesh) in [("112", [1, 1, 2]), ("222", [2, 2, 2])] {
        let kpts = cell.make_kpts(mesh).expect("kpts");
        let nk = kpts.len();
        let h = KptsHelper::new(&cell.a, &kpts);
        let map = h.symm_map.as_ref().expect("symm_map");
        assert_eq!(nk, scalar::<usize>(&out, &format!("symm_map_{tag}_nkpts")));
        assert_eq!(
            map.entries().len(),
            scalar::<usize>(&out, &format!("symm_map_{tag}_norbits")),
            "{tag}: orbit count"
        );
        let mut flat = Vec::new();
        for (key, members) in map.entries() {
            flat.extend(key.iter().map(|&x| x as f64));
            flat.push(members.len() as f64);
            for m in members {
                flat.extend(m.iter().map(|&x| x as f64));
            }
        }
        // Exact: these are integers, and the ORDER is the observable.
        assert_eq!(
            flat,
            block(&out, &format!("symm_map_{tag}")),
            "{tag}: orbits"
        );
        let mut ops = Vec::with_capacity(nk * nk * nk);
        for p in 0..nk {
            for q in 0..nk {
                for r in 0..nk {
                    ops.push(h.operation(p, q, r).expect("operation") as f64);
                }
            }
        }
        assert_eq!(
            ops,
            block(&out, &format!("operation_{tag}")),
            "{tag}: _operation"
        );
    }
}

// ---------------------------------------------------------------- 2. padding

#[test]
#[ignore = "requires PYSCF_ORACLE_VENV"]
fn padding_surface() {
    let Some(out) = rollup("padding") else {
        return;
    };
    // `kmp2.py:229-249`'s documented ragged example.
    let occ = vec![
        vec![2., 2., 0., 0., 0., 0.],
        vec![2., 2., 2., 0., 0., 0.],
        vec![2., 2., 0., 0., 0.],
    ];
    let variants = [
        FrozenK::Uniform(Frozen::None),
        FrozenK::Uniform(Frozen::Count(1)),
        FrozenK::Uniform(Frozen::List(vec![0, 1])),
        FrozenK::PerKpt(vec![vec![0], vec![0, 1], vec![1]]),
    ];
    for (i, frozen) in variants.iter().enumerate() {
        let nocc_k = get_nocc(&occ, frozen, true).expect("nocc/k").per_kpoint();
        let nmo_k = get_nmo(&occ, frozen, true).expect("nmo/k").per_kpoint();
        let mut flat: Vec<f64> = Vec::new();
        flat.extend(nocc_k.iter().map(|&x| x as f64));
        flat.extend(nmo_k.iter().map(|&x| x as f64));
        flat.push(get_nocc(&occ, frozen, false).expect("nocc").per_kpoint()[0] as f64);
        flat.push(get_nmo(&occ, frozen, false).expect("nmo").per_kpoint()[0] as f64);
        for m in get_frozen_mask(&occ, frozen).expect("mask") {
            flat.extend(m.iter().map(|&b| f64::from(u8::from(b))));
        }
        let PaddingIdx::Split { occupied, virtuals } =
            padding_k_idx(&nmo_k, &nocc_k, PaddingKind::Split).expect("split")
        else {
            unreachable!()
        };
        for x in occupied.iter().chain(&virtuals) {
            flat.push(x.len() as f64);
            flat.extend(x.iter().map(|&v| v as f64));
        }
        let PaddingIdx::Joint(joint) =
            padding_k_idx(&nmo_k, &nocc_k, PaddingKind::Joint).expect("joint")
        else {
            unreachable!()
        };
        for x in &joint {
            flat.push(x.len() as f64);
            flat.extend(x.iter().map(|&v| v as f64));
        }
        assert_eq!(
            flat,
            block(&out, &format!("padding_{i}")),
            "frozen variant {i}"
        );
    }
}

// ---------------------------------------------------------------- 3. ao2mo_7d

/// He/6-31g `[1,1,2]`, one k-quadruple, every `PeriodicDf` implementor, on
/// upstream's OWN randomly drawn MO coefficients — so the diff is the AO2MO
/// transform, not two independent SCFs.
#[test]
#[ignore = "requires PYSCF_ORACLE_VENV"]
fn ao2mo_and_ao2mo_7d() {
    let Some(out) = rollup("ao2mo7d") else {
        return;
    };
    let cell = common::helium_631g();
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let nao: usize = scalar(&out, "ao2mo7d_nao");
    assert_eq!(nao, cell.mol.nao_nr);
    let [ki, kj, kk, kl] = [
        scalar::<usize>(&out, "ki"),
        scalar::<usize>(&out, "kj"),
        scalar::<usize>(&out, "kk"),
        scalar::<usize>(&out, "kl"),
    ];
    let mos: Vec<MoCoeff> = (0..kpts.len())
        .map(|k| mo_from(&out, &format!("ao2mo7d_mo_{k}"), nao))
        .collect();
    let quad = [&mos[ki], &mos[kj], &mos[kk], &mos[kl]];

    let builders: Vec<(&str, Box<dyn PeriodicDf>)> = vec![
        (
            "fftdf",
            Box::new(Fftdf::with_mesh(cell.clone(), &kpts, [9, 9, 9]).expect("fftdf")),
        ),
        (
            "aftdf",
            Box::new(Aftdf::with_mesh(cell.clone(), &kpts, [9, 9, 9]).expect("aftdf")),
        ),
        ("gdf", Box::new(Gdf::new(cell.clone(), &kpts))),
        ("mdf", Box::new(Mdf::new(cell.clone(), &kpts))),
    ];
    for (name, df) in &builders {
        let got = df.ao2mo(quad, [ki, kj, kk, kl], false).expect("ao2mo");
        let want = cblock(&out, &format!("ao2mo_{name}"));
        let dev = max_dev(&got.restore_s1().data, &want);

        let seven = df
            .ao2mo_7d([&mos, &mos, &mos, &mos], 1.0)
            .expect("ao2mo_7d");
        let want7 = cblock(&out, &format!("ao2mo7d_{name}"));
        let off = seven.block_offset(ki, kj, kk);
        let len = seven.block_len();
        let slot = CTensor::from_planes(
            seven.data.re[off..off + len].to_vec(),
            seven.data.im[off..off + len].to_vec(),
        );
        let dev7 = max_dev(&slot, &want7);

        // The AO-level block, so a residual is attributable: an equal AO and MO
        // deviation is the INTEGRAL, not the transform.
        let ao = df.get_ao_eri([ki, kj, kk, kl], false).expect("get_ao_eri");
        let want_ao = cblock(&out, &format!("aoeri_{name}"));
        let dev_ao = max_dev(&ao.restore_s1().data, &want_ao);
        println!(
            "[{name}] ao_eri_max_dev={dev_ao:.3e} ao2mo_max_dev={dev:.3e} \
             ao2mo_7d_max_dev={dev7:.3e}"
        );

        // The two PLANE-WAVE builders are gated: they compute the same integral
        // the same way, so anything above round-off is a defect. The two
        // GAUSSIAN builders are REPORTED, not gated — their residual is the
        // inherited `j3c` baseline gap that `15-VERIFICATION.md` row 4 records
        // and assigns to Phase 14, and gating it here would either hide that
        // finding behind a loosened number or fail for a reason Phase 15 does
        // not own. The internal route agreement IS gated, in `tests/kmp2.rs`,
        // at 2e-15 on the same mean field.
        if *name == "fftdf" {
            assert!(dev < FFT_AO2MO_TOL, "[ao2mo {name}] {dev:.3e}");
            assert!(dev7 < FFT_AO2MO_TOL, "[ao2mo_7d {name}] {dev7:.3e}");
        } else if *name == "aftdf" {
            // AFTDF's residual is its ANALYTIC AO integral, not the transform:
            // `ao_eri_max_dev` and `ao2mo_max_dev` are the same number, and the
            // port's own AO-first and MO-first routes agree to 1.8e-16
            // (`pbc_ao2mo_mofirst.rs`). Gated at the measured value so a
            // TRANSFORM regression still trips it.
            assert!(dev < AFT_AO2MO_TOL, "[ao2mo {name}] {dev:.3e}");
            assert!(dev7 < AFT_AO2MO_TOL, "[ao2mo_7d {name}] {dev7:.3e}");
        } else {
            assert!(
                dev.is_finite() && dev7.is_finite(),
                "[{name}] produced a non-finite element"
            );
        }
    }
}

/// MEASURED on He/6-31g `[1,1,2]`, mesh 9, upstream's own random MO draw:
/// FFTDF `5.793e-12`. One order of headroom.
const FFT_AO2MO_TOL: f64 = 6e-11;

/// MEASURED the same way: AFTDF `1.891e-4`, on a block whose largest element is
/// `33.6` — a `5.6e-6` RELATIVE difference in the analytic `ft_ao` integral at
/// this deliberately coarse mesh, identical at the AO and MO levels. See
/// `15-VERIFICATION.md §3.4`.
const AFT_AO2MO_TOL: f64 = 6e-4;

// ---------------------------------------------------------------- 4. Lov

#[test]
#[ignore = "requires PYSCF_ORACLE_VENV and a diamond GDF build"]
fn lov_blocks() {
    let Some(out) = rollup("lov") else {
        return;
    };
    let cell = common::diamond_anchor();
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let nk: usize = scalar(&out, "lov_nkpts");
    let nocc: usize = scalar_on(&out, "lov_nkpts", "nocc");
    let nvir: usize = scalar_on(&out, "lov_nkpts", "nvir");
    assert_eq!(nk, kpts.len());
    let nao = cell.mol.nao_nr;
    let mos: Vec<MoCoeff> = (0..nk)
        .map(|k| mo_from(&out, &format!("lov_mo_{k}"), nao))
        .collect();
    let df = Gdf::new(cell, &kpts);
    let table = build_lov(&df, &mos, nocc).expect("Lov");
    assert_eq!(table.nocc, nocc);
    assert_eq!(table.nvir, nvir);

    let naux = block(&out, "lov_naux");
    let mut worst = 0.0f64;
    for ki in 0..nk {
        for kj in 0..nk {
            let (n, got) = table.block(ki, kj);
            assert_eq!(
                *n as f64,
                naux[ki * nk + kj],
                "Lov naux differs at ({ki},{kj})"
            );
            // Upstream stores (naux, nocc, nvir); this port stores `L` fastest.
            let want = cblock(&out, &format!("lov_{ki}_{kj}"));
            let mut reordered = CTensor::zeros(want.len());
            for l in 0..*n {
                for i in 0..nocc {
                    for a in 0..nvir {
                        let src = (i * nvir + a) * *n + l;
                        let dst = (l * nocc + i) * nvir + a;
                        reordered.re[dst] = got.re[src];
                        reordered.im[dst] = got.im[src];
                    }
                }
            }
            worst = worst.max(max_dev(&reordered, &want));
        }
    }
    // MEASURED, not asserted cold: the residual here is the inherited GDF
    // `j3c` baseline (15-VERIFICATION row 4, owned by Phase 14), not the Lov
    // transform — which `tests/kmp2.rs` pins against the four-index route at
    // 2e-15 on the same mean field.
    println!("[lov diamond/gth-szv [1,1,2]] max_dev={worst:.6e}");
    assert!(worst.is_finite(), "Lov produced a non-finite element");
}

// ---------------------------------------------------------------- 5. KMP2

#[test]
#[ignore = "requires PYSCF_ORACLE_VENV and four periodic SCFs"]
fn kmp2_energies() {
    let Some(out) = rollup("kmp2") else {
        return;
    };
    for (tag, cell) in [
        ("diamond", common::diamond_anchor()),
        ("helium", common::helium_631g()),
    ] {
        let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
        for route in ["fftdf", "gdf"] {
            let df: Box<dyn PeriodicDf> = if route == "fftdf" {
                Box::new(Fftdf::new(cell.clone(), &kpts).expect("fftdf"))
            } else {
                Box::new(Gdf::new(cell.clone(), &kpts))
            };
            let mut mf = Krhf::from_df(df);
            mf.exxdiv = None;
            let mut cfg = KScfConfig::for_cell(mf.cell());
            cfg.conv_tol = 1e-11;
            let result = mf.kernel(&cfg).expect("SCF");
            assert!(result.converged, "{tag}/{route}: SCF did not converge");
            let mut mp = Kmp2::new(&result, mf.with_df.as_ref()).expect("KMP2");
            mp.with_t2 = false;
            let got = mp.kernel().expect("KMP2");
            let key = format!("kmp2_{tag}_{route}");
            let want: f64 = scalar_on(&out, &key, "e_corr");
            let want_ss: f64 = scalar_on(&out, &key, "ss");
            let want_os: f64 = scalar_on(&out, &key, "os");
            // The mean-field energy travels with it: a KMP2 residual on a DF
            // route means nothing until you know whether its SCF agreed.
            let want_hf: f64 = scalar_on(&out, &key, "e_hf");
            println!(
                "[{key}] e_hf_rust={:.17} e_hf_upstream={want_hf:.17} \
                 e_hf_residual={:.3e}",
                result.e_tot,
                (result.e_tot - want_hf).abs()
            );
            println!(
                "[{key}] rust={:.17} upstream={want:.17} residual={:.3e} \
                 ss_residual={:.3e} os_residual={:.3e}",
                got.e_corr,
                (got.e_corr - want).abs(),
                (got.e_corr_ss - want_ss).abs(),
                (got.e_corr_os - want_os).abs()
            );
            // `e_corr_ss + e_corr_os == e_corr` exactly, on every route.
            assert_eq!(got.e_corr_ss + got.e_corr_os, got.e_corr);
            if route == "fftdf" {
                // The measured phase gate (`measurements/README.md`). GDF is
                // reported, not gated: 15-VERIFICATION row 4 records the
                // inherited baseline gap and assigns it to Phase 14.
                assert!(
                    (got.e_corr - want).abs() < 2e-6,
                    "[{key}] residual {:.3e}",
                    (got.e_corr - want).abs()
                );
                assert!((got.e_corr_ss - want_ss).abs() < 2e-6);
                assert!((got.e_corr_os - want_os).abs() < 2e-6);
            }
        }
    }
}

// ---------------------------------------------------------------- 6. t2/RDM

#[test]
#[ignore = "requires PYSCF_ORACLE_VENV and a periodic SCF"]
fn t2_rdm1_and_gamma1() {
    let Some(out) = rollup("t2rdm") else {
        return;
    };
    let cell = common::helium_631g();
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let mut mf = Krhf::from_df(Box::new(
        Fftdf::with_mesh(cell.clone(), &kpts, [9, 9, 9]).expect("fftdf"),
    ));
    mf.exxdiv = None;
    let mut cfg = KScfConfig::for_cell(mf.cell());
    cfg.conv_tol = 1e-11;
    let result = mf.kernel(&cfg).expect("SCF");
    let mp = Kmp2::new(&result, mf.with_df.as_ref()).expect("KMP2");
    let r = mp.kernel().expect("KMP2");
    let t2 = r.t2.as_ref().expect("T2");
    let nk = t2.nkpts;
    assert_eq!(nk, scalar::<usize>(&out, "nkpts"));
    assert_eq!(t2.nocc, scalar::<usize>(&out, "nocc"));

    // Upstream's t2 is (nk, nk, nk, nocc, nocc, nvir, nvir) at [ki, kj, ka].
    let want = cblock(&out, "t2");
    let per = t2.nocc * t2.nocc * t2.nvir * t2.nvir;
    let mut worst = 0.0f64;
    for ki in 0..nk {
        for kj in 0..nk {
            for ka in 0..nk {
                let off = ((ki * nk + kj) * nk + ka) * per;
                let slice = CTensor::from_planes(
                    want.re[off..off + per].to_vec(),
                    want.im[off..off + per].to_vec(),
                );
                worst = worst.max(max_dev(t2.block(ki, kj, ka), &slice));
            }
        }
    }
    println!("[t2 helium/6-31g [1,1,2]] max_dev={worst:.3e}");
    assert!(worst < 2e-8, "t2 max_dev={worst:.3e}");

    for (kind, name) in [
        (RdmKind::Padded, "rdm1_padded"),
        (RdmKind::Compact, "rdm1_compact"),
    ] {
        let got = mp.make_rdm1(t2, kind).expect("RDM1");
        let want = cblock(&out, name);
        let per = want.len() / nk;
        let mut worst = 0.0f64;
        for k in 0..nk {
            let slice = CTensor::from_planes(
                want.re[k * per..(k + 1) * per].to_vec(),
                want.im[k * per..(k + 1) * per].to_vec(),
            );
            worst = worst.max(max_dev(&got[k], &slice));
        }
        println!("[{name}] max_dev={worst:.3e}");
        assert!(worst < 2e-8, "{name} max_dev={worst:.3e}");
    }

    let (doo, dvv) = gamma1_intermediates(t2, &mp.khelper.kconserv);
    for (got, name) in [(&doo, "gamma1_doo"), (&dvv, "gamma1_dvv")] {
        let want = cblock(&out, name);
        let per = want.len() / nk;
        let mut worst = 0.0f64;
        for k in 0..nk {
            let slice = CTensor::from_planes(
                want.re[k * per..(k + 1) * per].to_vec(),
                want.im[k * per..(k + 1) * per].to_vec(),
            );
            worst = worst.max(max_dev(&got[k], &slice));
        }
        println!("[{name}] max_dev={worst:.3e}");
        assert!(worst < 2e-8, "{name} max_dev={worst:.3e}");
    }
}

// ---------------------------------------------------------------- 7. stagger

#[test]
#[ignore = "requires PYSCF_ORACLE_VENV and three periodic SCFs"]
fn stagger_energies() {
    let script = ".planning/phases/15-periodic-ao2mo-kmp2/measurements/stagger.py";
    let Some(out) = run(&["-u", script]) else {
        return;
    };
    // The committed Rust gate values in `tests/kmp2_stagger.rs` must still be
    // what upstream produces.
    for (key, want) in [
        ("stagger_submesh_fftdf", -0.016089900380356827),
        ("stagger_fullmesh_fftdf", -0.014028716824109303),
        ("standard_kmp2_fftdf", -0.014390203713094872),
    ] {
        let got: f64 = scalar(&out, key);
        println!("[{key}] upstream={got:.17} committed={want:.17}");
        assert!(
            (got - want).abs() < 1e-12,
            "[{key}] upstream drifted: {got} vs {want}"
        );
    }
    assert_eq!(scalar::<usize>(&out, "stagger_submesh_nkpts_ov"), 1);
    assert_eq!(scalar::<usize>(&out, "stagger_fullmesh_nkpts_ov"), 8);
}

// ---------------------------------------------------------------- 8. KUMP2

#[test]
#[ignore = "requires PYSCF_ORACLE_VENV"]
fn upstream_kump2_kernel_remains_an_explicit_refusal() {
    let code = "import inspect,pyscf; from pyscf.pbc.mp import kump2; assert pyscf.__version__=='2.12.1'; print('pyscf.__version__=2.12.1'); s=inspect.getsource(kump2.KUMP2.kernel); assert 'raise NotImplementedError' in s; print('KUMP2 refusal confirmed')";
    let Some(out) = run(&["-c", code]) else {
        return;
    };
    assert!(out.contains("refusal confirmed"));
}

// ---------------------------------------------------------------- 9. MO-first

/// The route the phase's own FFTDF anchor runs on, every conserving quadruple.
///
/// **FFTDF only, deliberately.** `Aftdf::ao2mo` dispatches to the same
/// `aft_general_mo_first`, and [`ao2mo_and_ao2mo_7d`] already gates that
/// against upstream — on He/6-31g mesh 9, where the analytic `ft_ao` costs
/// seconds. On diamond's 47^3 mesh a single AFT k-quadruple ran past 19
/// CPU-minutes without finishing, on BOTH sides of the comparison, and buys
/// nothing the cheaper fixture does not already prove.
#[test]
#[ignore = "requires PYSCF_ORACLE_VENV and a diamond SCF"]
fn mo_first_ao2mo_block() {
    use pyscf_pbc_df::pbc_ao2mo::fft_general_mo_first;

    let Some(out) = rollup("mofirst") else {
        return;
    };
    let cell = common::diamond_anchor();
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let nk: usize = scalar(&out, "mofirst_nkpts");
    let nocc: usize = scalar_on(&out, "mofirst_nkpts", "nocc");
    let nvir: usize = scalar_on(&out, "mofirst_nkpts", "nvir");
    assert_eq!(nk, kpts.len());
    let nao = cell.mol.nao_nr;
    let mos: Vec<MoCoeff> = (0..nk)
        .map(|k| mo_from(&out, &format!("mofirst_mo_{k}"), nao))
        .collect();
    let occ: Vec<MoCoeff> = mos.iter().map(|m| slice_mo(m, 0, nocc)).collect();
    let vir: Vec<MoCoeff> = mos.iter().map(|m| slice_mo(m, nocc, nocc + nvir)).collect();

    let fft = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    // The two sides must be on the same FFT mesh or this compares two
    // different integrals. Upstream's `df.FFTDF(cell, kpts)` takes it from the
    // cell, and so does `Fftdf::new` — asserted rather than assumed.
    let mesh_line = format!(
        "mofirst_mesh=[{}, {}, {}]",
        fft.mesh[0], fft.mesh[1], fft.mesh[2]
    );
    assert!(
        out.contains(&mesh_line),
        "mesh mismatch: this port is on {:?}",
        fft.mesh
    );
    println!("[mofirst] mesh={:?}", fft.mesh);
    let kc = pyscf_pbc_lib::kpts_helper::get_kconserv(&cell.a, &kpts);
    let mut worst = 0.0f64;
    for ki in 0..nk {
        for ka in 0..nk {
            for kj in 0..nk {
                let kb = kc.get(ki, ka, kj) as usize;
                let k4 = [kpts[ki], kpts[ka], kpts[kj], kpts[kb]];
                let quad = [&occ[ki], &vir[ka], &occ[kj], &vir[kb]];
                let want = cblock(&out, &format!("mofirst_fftdf_{ki}_{ka}_{kj}"));
                let got = fft_general_mo_first(&fft, quad, k4, None).expect("fft MO-first");
                worst = worst.max(max_dev(&got.data, &want));
            }
        }
    }
    println!(
        "[mofirst diamond/gth-szv [1,1,2]] fft_max_dev={worst:.3e} over {} quadruples",
        nk * nk * nk
    );
    // MEASURED `5.815e-14` over all eight quadruples. This comparison is NOT
    // cross-SCF — it runs on upstream's own padded MO coefficients — so its
    // floor is pure arithmetic and the `2e-8` used for the cross-SCF rows
    // would be six orders too loose here.
    assert!(worst < 2e-12, "FFT MO-first max_dev={worst:.3e}");
}

fn slice_mo(m: &MoCoeff, lo: usize, hi: usize) -> MoCoeff {
    let n = hi - lo;
    let mut c = CTensor::zeros(m.nao * n);
    for p in 0..m.nao {
        for (j, q) in (lo..hi).enumerate() {
            c.re[p * n + j] = m.c.re[p * m.nmo + q];
            c.im[p * n + j] = m.c.im[p * m.nmo + q];
        }
    }
    MoCoeff::new(m.nao, n, c)
}

// ------------------------------------------------- the committed anchor

/// `measurements/anchor.py` reproduces `kmp2.py:795-821`. Kept as its own test
/// so a drift in upstream's own constant is separable from a drift in this
/// port.
#[test]
#[ignore = "requires PYSCF_ORACLE_VENV and a full periodic SCF"]
fn committed_diamond_anchor_is_still_reproducible() {
    let script = ".planning/phases/15-periodic-ao2mo-kmp2/measurements/anchor.py";
    let Some(out) = run(&["-u", script]) else {
        return;
    };
    let value: f64 = scalar(&out, "e_corr");
    println!(
        "[anchor] upstream={value:.17} source_constant=-0.204721432828996 residual={:.3e}",
        (value - -0.204721432828996).abs()
    );
    assert!((value - -0.204721432828996).abs() < 2e-6, "{value}");
}

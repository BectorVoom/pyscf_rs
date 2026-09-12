//! **The pointwise XC oracle: `libxc_rs` against PySCF's C libxc, bit for bit.**
//!
//! # Why this file exists when `xc_eval_bitexact.rs` already does
//!
//! `xc_eval_bitexact.rs` is named for bit-exactness but **PySCF is not in
//! it**: it checks LDA_X against an ANALYTIC Slater formula at `1e-12`
//! relative, checks that PBE reduces to Slater at `sigma = 0`, and checks
//! xcfun against libxc. All useful, none of them the question "does this
//! agree with the library PySCF actually calls".
//!
//! This file asks that question directly: identical `(rho, sigma)` in, and
//! `to_bits()` on the way out.
//!
//! # The two conventions that have to line up, or the comparison is noise
//!
//! 1. **`exc` is per-particle upstream, energy-density here.**
//!    `pyscf.dft.libxc.eval_xc` returns `zk` (energy per particle);
//!    `XcBackend::eval` returns `(Σ fac · zk) · rho`. The oracle therefore
//!    multiplies by `rho` ON THE PYTHON SIDE, so both sides perform the same
//!    `zk * rho` product and a bit difference means `zk` differed — not that
//!    the multiplication was done in a different order. Open-shell uses
//!    `rho_a + rho_b` on both sides.
//!
//! 2. **Upstream takes `∇rho`, not `sigma`.** `eval_xc` computes
//!    `sigma = einsum('ip,ip->p', drho, drho)` internally, so handing it a
//!    `sigma` directly is impossible. The fixture instead picks `g` and sets
//!    `drho = (g, 0, 0)` upstream and `sigma = g * g` here: upstream's
//!    `g*g + 0*0 + 0*0` is exactly `g*g` in f64, so both sides see the SAME
//!    `sigma` bits. Passing `sqrt(sigma)` would not — the round trip through
//!    `sqrt` then squaring is not the identity.
//!
//! # Corpus
//!
//! Single-component `slater,` / `,vwn` / `pbe,` / `,pbe` isolate the
//! functional kernels. The compounds `pbe,pbe` / `svwn` / `blyp` / `b3lyp` /
//! `pbe0` add the combination step: PySCF's `merge_xc`
//! (`lib/dft/libxc_itrf.c`) sums `fac * zk` per PARTICLE across the
//! components and Python multiplies by `rho` afterwards. `XcBackend::eval`
//! does the same. Scaling each component by `rho` inside the loop rounds
//! differently and was 1 ULP off on `pbe,pbe` `exc`.
//!
//! # MEASURED 2026-09-11: every quantity bit-identical
//!
//! All nine codes, every point, `exc` / `vrho` / `vsigma`: **0 ULP**, and
//! the same for the open-shell corpus (`libxc_rs_matches_pyscf_libxc_pointwise_uks`,
//! which goes through `eval_uks`, the interleaved polarized buffers, and every
//! spin channel). The assertions are exact. Reaching it took libxc_rs matching a GCC -O3 build of
//! libxc on glibc, specifically:
//!
//! * glibc's own (not correctly rounded) `cbrt` at runtime;
//! * constant-argument libm calls folded correctly rounded, as GCC does
//!   through MPFR;
//! * gcc-exact `M_CBRT*` / `X2S` constants.
//!
//! On the pyscf-dft side it took the `merge_xc` order above and a lossless
//! JSON float parse (`serde_json`'s `float_roundtrip`).
//!
//! # History: the number that looked like a disaster (2026-09-10)
//!
//! Before those fixes the single-component worst was 4 ULP, and the
//! compound `pbe,pbe` `vsigma` reported **2.25e15 ULP** at
//! `rho = 1e-6, sigma = 0`, and that figure is REAL but not a defect —
//! working it out is the whole reason `Cmp` carries the raw values:
//!
//! * the two components' `vsigma` there are ~`+1.3e5` and ~`-1.3e5`;
//! * they cancel to a result of ~`1e-10` — a cancellation of ~1e15;
//! * the absolute disagreement is `5.8207660913467410e-11`, which is
//!   **exactly 2 ULP of a summand in `[2^17, 2^18)`** (`2 · 2^-35`);
//! * so the result's ~50 % RELATIVE error is entirely inherited rounding of
//!   the inputs to a catastrophic cancellation, not a functional difference.
//!
//! `pbe,` alone reports the same `5.821e-11` worst-absolute, which is the
//! same summand rounding before any cancellation — that coincidence is the
//! confirmation. **A ULP count is meaningless across a cancellation**;
//! judge that point on the absolute number.

#![cfg(feature = "libxc")]

use pyscf_dft::parser::libxc;
use pyscf_dft::xc_backend::{DerivOrder, RhoBlock, XcBackend};
use std::path::PathBuf;
use std::process::Command;

/// Densities spanning the range a real grid sees: core-like down to tail.
const RHO: [f64; 8] = [1e-6, 1e-3, 0.05, 0.1, 0.5, 1.0, 3.7, 42.0];

/// `|∇rho|` values. `sigma = g * g` on both sides — see the module doc.
const GRAD: [f64; 8] = [0.0, 1e-4, 0.01, 0.1, 0.37, 1.0, 2.5, 11.0];

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// `PYSCF_ORACLE_VENV` -> interpreter, or `None` (caller SKIPS, never fails).
fn oracle_python() -> Option<PathBuf> {
    let raw = std::env::var("PYSCF_ORACLE_VENV").ok()?.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    let p = if matches!(raw.as_str(), "1" | "true" | "auto" | "yes") {
        workspace_root().join(".venv/bin/python")
    } else {
        let c = PathBuf::from(&raw);
        if c.is_dir() { c.join("bin/python") } else { c }
    };
    assert!(p.exists(), "PYSCF_ORACLE_VENV -> {p:?} does not exist");
    Some(p)
}

const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
import pyscf
from pyscf.dft import libxc

code = sys.argv[1]
rho  = np.array(json.loads(sys.argv[2]), dtype=np.float64)
grad = np.array(json.loads(sys.argv[3]), dtype=np.float64)
fam  = sys.argv[4]

if fam == 'lda':
    inp = rho
else:
    # drho = (g, 0, 0) so upstream's sigma = g*g + 0*0 + 0*0 == g*g exactly.
    inp = np.zeros((4, rho.size))
    inp[0] = rho
    inp[1] = grad

exc, vxc, _fxc, _kxc = libxc.eval_xc(code, inp, spin=0, relativity=0, deriv=1)

# exc is PER PARTICLE upstream; multiply here so both sides do the same
# zk * rho product (see the module doc).
out = {
    "pyscf_version": pyscf.__version__,
    "exc":  (np.asarray(exc) * rho).tolist(),
    "vrho": np.asarray(vxc[0]).tolist(),
}
out["vsigma"] = None if (fam == 'lda' or vxc[1] is None) else np.asarray(vxc[1]).tolist()
out["exc_raw"] = np.asarray(exc).tolist()
# Which shared libraries actually served this evaluation. Under `cargo test`
# the child inherits cargo's LD_LIBRARY_PATH, which OVERRIDES a RUNPATH, so
# the libxc PySCF loads here is not guaranteed to be the one it ships.
out["loaded_xc"] = sorted({l.split()[-1] for l in open('/proc/self/maps') if 'xc' in l.split()[-1] and l.split()[-1].endswith(tuple(['.so'] + [f'.so.{i}' for i in range(20)]))} | {l.split()[-1] for l in open('/proc/self/maps') if l.split()[-1].split('/')[-1].startswith(('libxc', 'libm.'))})
print(json.dumps(out))
"#;

fn run_python(py: &PathBuf, args: &[String]) -> serde_json::Value {
    run_script(py, ORACLE_PY, args)
}

fn run_script(py: &PathBuf, script: &str, args: &[String]) -> serde_json::Value {
    // Unique per CALL: tests share a process, so a pid-only name races when
    // the harness runs them on parallel threads.
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path =
        std::env::temp_dir().join(format!("libxc_pointwise_{}_{seq}.py", std::process::id()));
    std::fs::write(&path, script).expect("write oracle script");
    let root = workspace_root();
    let out = Command::new(py)
        .arg(&path)
        .args(args)
        .env("PYTHONPATH", &root)
        .current_dir(&root)
        .output()
        .expect("spawn upstream python");
    let _ = std::fs::remove_file(&path);
    assert!(
        out.status.success(),
        "oracle failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let s = String::from_utf8_lossy(&out.stdout);
    let line = s
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .unwrap_or_else(|| panic!("oracle produced no JSON:\n{s}"));
    serde_json::from_str(line).expect("oracle JSON parses")
}

/// One ULP at `x` — the gap to the next representable f64.
fn ulp(x: f64) -> f64 {
    let a = x.abs();
    if a == 0.0 {
        return f64::MIN_POSITIVE;
    }
    f64::from_bits(a.to_bits() + 1) - a
}

/// Bit-identical count and worst deviation in ULP, for one quantity.
///
/// **`worst_ulp` alone is a trap near zero** and the raw values are carried
/// alongside it for that reason. A ULP count divides by the spacing at the
/// operands' own magnitude, so comparing `1e-30` against `0.0` reports ~1e16
/// ULP while the ABSOLUTE difference is 1e-30 — physically nothing. The
/// report prints `got`/`want` at the worst point so the reader can tell a
/// real disagreement from a denormal-scale artefact instead of guessing.
struct Cmp {
    n: usize,
    exact: usize,
    worst_ulp: f64,
    worst_at: usize,
    worst_got: f64,
    worst_want: f64,
    /// The largest ABSOLUTE difference, which near zero is the honest number.
    worst_abs: f64,
}

fn compare(got: &[f64], want: &[f64]) -> Cmp {
    assert_eq!(got.len(), want.len(), "length mismatch vs upstream");
    let mut c = Cmp {
        n: got.len(),
        exact: 0,
        worst_ulp: 0.0,
        worst_at: 0,
        worst_got: 0.0,
        worst_want: 0.0,
        worst_abs: 0.0,
    };
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if g.to_bits() == w.to_bits() {
            c.exact += 1;
            continue;
        }
        c.worst_abs = c.worst_abs.max((g - w).abs());
        let u = (g - w).abs() / ulp(g.abs().max(w.abs()));
        if u > c.worst_ulp {
            c.worst_ulp = u;
            c.worst_at = i;
            c.worst_got = g;
            c.worst_want = w;
        }
    }
    c
}

fn pull(v: &serde_json::Value, key: &str) -> Option<Vec<f64>> {
    v[key].as_array().map(|a| {
        a.iter()
            .map(|x| x.as_f64().expect("f64 in oracle payload"))
            .collect()
    })
}

/// One quantity's line in the report.
fn report(name: &str, c: &Cmp) {
    if c.exact == c.n {
        println!("    {name:<7} {}/{} BIT-IDENTICAL", c.exact, c.n);
        return;
    }
    println!(
        "    {name:<7} {}/{} bit-identical   worst {:.3} ULP / {:.3e} abs \
         (point {}, rho={:e})\n              got  {:.17e}\n              want {:.17e}",
        c.exact,
        c.n,
        c.worst_ulp,
        c.worst_abs,
        c.worst_at,
        RHO[c.worst_at],
        c.worst_got,
        c.worst_want,
    );
}

/// Evaluate one functional both ways and report. Returns the worst ULP over
/// every quantity compared.
fn measure(py: &PathBuf, code: &str, gga: bool) -> f64 {
    let spec = libxc::parse_xc(code).unwrap_or_else(|e| panic!("{code} parses: {e:?}"));
    let ncomp = spec.components().len();

    let sigma: Vec<f64> = GRAD.iter().map(|&g| g * g).collect();
    let rb = if gga {
        RhoBlock::Gga {
            rho: &RHO,
            sigma: &sigma,
        }
    } else {
        RhoBlock::Lda { rho: &RHO }
    };
    let got = XcBackend::Libxc
        .eval(&spec, &rb, DerivOrder::Vxc)
        .unwrap_or_else(|e| panic!("{code} eval: {e:?}"));

    let up = run_python(
        py,
        &[
            code.to_string(),
            serde_json::to_string(&RHO.to_vec()).expect("json"),
            serde_json::to_string(&GRAD.to_vec()).expect("json"),
            if gga { "gga".into() } else { "lda".into() },
        ],
    );

    let c_exc = compare(&got.exc, &pull(&up, "exc").expect("upstream exc"));
    let c_vrho = compare(&got.vrho, &pull(&up, "vrho").expect("upstream vrho"));
    let c_vsig = pull(&up, "vsigma").map(|w| compare(&got.vsigma, &w));

    println!(
        "  {code}  ({ncomp} libxc component(s), {} points)",
        RHO.len()
    );
    report("exc", &c_exc);
    report("vrho", &c_vrho);
    let mut worst = c_exc.worst_ulp.max(c_vrho.worst_ulp);
    if let Some(c) = &c_vsig {
        report("vsigma", c);
        worst = worst.max(c.worst_ulp);
    }
    worst
}

/// The measurement. **Reports first, asserts second** — the assertion is a
/// loose regression bound (a few ULP), because "bit-exact" is the thing being
/// MEASURED here, not the thing being assumed. If every quantity comes back
/// `n/n bit-identical`, the log says so and the bound can be tightened to
/// `worst == 0.0` in a follow-up with the evidence to justify it.
#[test]
fn libxc_rs_matches_pyscf_libxc_pointwise() {
    let Some(py) = oracle_python() else {
        eprintln!("PYSCF_ORACLE_VENV unset — skipping the pointwise libxc oracle");
        return;
    };

    println!("=== libxc_rs vs PySCF's C libxc, pointwise ===");
    let mut worst = 0.0f64;
    for (code, gga) in [
        ("slater,", false),
        (",vwn", false),
        ("pbe,", true),
        (",pbe", true),
    ] {
        worst = worst.max(measure(&py, code, gga));
    }
    println!("SINGLE-COMPONENT worst deviation: {worst:.3} ULP");

    // Compound: the components are summed per particle in `merge_xc`'s order
    // and scaled by `rho` once, so these must be exact too.
    println!("--- compound ---");
    let mut compound = 0.0f64;
    for (code, gga) in [
        ("pbe,pbe", true),
        ("svwn", false),
        ("blyp", true),
        ("b3lyp", true),
        ("pbe0", true),
    ] {
        compound = compound.max(measure(&py, code, gga));
    }
    println!("COMPOUND worst deviation: {compound:.3} ULP");

    assert!(
        worst == 0.0 && compound == 0.0,
        "libxc_rs is no longer bit-identical to PySCF's C libxc: single-component \
         worst {worst:.3} ULP, compound worst {compound:.3} ULP (see the log above for \
         the quantity and point)"
    );
}

/// **Layer attribution.** Evaluates the functional through `libxc_rs`
/// directly -- `BatchEvaluator`, `DerivativeOrder::Vxc`, exactly what
/// `XcBackend::Libxc` does for one component -- inside THIS build (same
/// profile, flags and crate graph as production), and prints, per point,
/// whether each layer's bits match upstream:
///
/// * `raw zk`  -- `libxc_rs`'s per-particle `zk` vs upstream's `exc`
/// * `raw vrho`-- `libxc_rs`'s `vrho` vs upstream's
/// * `backend vrho` -- `XcBackend::eval`'s `vrho` vs `libxc_rs`'s raw one
///
/// A standalone replica of a kernel can agree with upstream while this
/// build does not; this is the test that says which layer diverges.
/// Report-only.
#[test]
fn layer_attribution_lda() {
    use libxc_rs::{
        BatchEvaluator, DerivativeOrder, Functional, FunctionalId, LdaInput, LdaOutput, Spin,
    };
    let Some(py) = oracle_python() else {
        eprintln!("PYSCF_ORACLE_VENV unset — skipping");
        return;
    };
    let np = RHO.len();
    for code in ["slater,", ",vwn"] {
        let spec = libxc::parse_xc(code).expect("parses");
        assert_eq!(
            spec.components().len(),
            1,
            "{code}: single component expected"
        );
        let (id, fac) = spec.components()[0];
        assert_eq!(fac, 1.0);
        let f = Functional::new(
            FunctionalId::from_raw(id as u16).unwrap(),
            Spin::Unpolarized,
        )
        .unwrap();
        let mut batch = BatchEvaluator::new(Spin::Unpolarized, np);
        let input = LdaInput::new(&RHO, np, Spin::Unpolarized).unwrap();
        let (mut zk, mut vrho) = (vec![0.0; np], vec![0.0; np]);
        {
            let mut o = LdaOutput::new(
                Some(&mut zk),
                Some(&mut vrho),
                None,
                None,
                None,
                np,
                Spin::Unpolarized,
            )
            .unwrap();
            batch
                .evaluate(&f, &input, DerivativeOrder::Vxc, &mut o)
                .unwrap();
        }
        let be = XcBackend::Libxc
            .eval(&spec, &RhoBlock::Lda { rho: &RHO }, DerivOrder::Vxc)
            .unwrap();
        let up = run_python(
            &py,
            &[
                code.to_string(),
                serde_json::to_string(&RHO.to_vec()).unwrap(),
                serde_json::to_string(&GRAD.to_vec()).unwrap(),
                "lda".into(),
            ],
        );
        let (uz, uv) = (pull(&up, "exc_raw").unwrap(), pull(&up, "vrho").unwrap());
        println!("=== {code} (libxc id {id}) ===");
        println!("  upstream loaded: {}", up["loaded_xc"]);
        for i in 0..np {
            let m = |a: f64, b: f64| if a.to_bits() == b.to_bits() { "=" } else { "X" };
            println!(
                "  rho={:<7e} raw zk {} ({:#018x} vs {:#018x})  raw vrho {} ({:#018x} vs {:#018x})  backend vrho {} ",
                RHO[i],
                m(zk[i], uz[i]),
                zk[i].to_bits(),
                uz[i].to_bits(),
                m(vrho[i], uv[i]),
                vrho[i].to_bits(),
                uv[i].to_bits(),
                m(be.vrho[i], vrho[i])
            );
        }
    }
}

const ORACLE_UKS_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.dft import libxc

code = sys.argv[1]
ra, rb, ga, gb = (np.array(json.loads(a), dtype=np.float64) for a in sys.argv[2:6])
fam = sys.argv[6]

if fam == 'lda':
    inp = (ra, rb)
else:
    # drho_s = (g_s, 0, 0): sigma_aa = ga*ga, sigma_ab = ga*gb, sigma_bb = gb*gb
    # exactly, the products the Rust side forms.
    a = np.zeros((4, ra.size)); a[0] = ra; a[1] = ga
    b = np.zeros((4, rb.size)); b[0] = rb; b[1] = gb
    inp = (a, b)

exc, vxc, _fxc, _kxc = libxc.eval_xc(code, inp, spin=1, relativity=0, deriv=1)
vr = np.asarray(vxc[0])          # (N, 2), measured
out = {
    "exc": (np.asarray(exc) * (ra + rb)).tolist(),
    "vrho_a": vr[:, 0].tolist(),
    "vrho_b": vr[:, 1].tolist(),
}
if fam != 'lda':
    vs = np.asarray(vxc[1])      # (N, 3): aa, ab, bb
    out["vsigma_aa"] = vs[:, 0].tolist()
    out["vsigma_ab"] = vs[:, 1].tolist()
    out["vsigma_bb"] = vs[:, 2].tolist()
print(json.dumps(out))
"#;

/// The open-shell counterpart: `XcBackend::eval_uks` against
/// `eval_xc(..., spin=1)`, every spin channel, bit for bit. `rho_a = RHO`,
/// `rho_b = 0.4 * RHO` (a real spin polarisation at every point, so the
/// alpha/beta interleave cannot hide a swap), `|∇rho_b| = 0.7 * |∇rho_a|`.
#[test]
fn libxc_rs_matches_pyscf_libxc_pointwise_uks() {
    let Some(py) = oracle_python() else {
        eprintln!("PYSCF_ORACLE_VENV unset — skipping the open-shell pointwise libxc oracle");
        return;
    };
    let rho_b: Vec<f64> = RHO.iter().map(|&r| 0.4 * r).collect();
    let grad_b: Vec<f64> = GRAD.iter().map(|&g| 0.7 * g).collect();
    let saa: Vec<f64> = GRAD.iter().map(|&g| g * g).collect();
    let sab: Vec<f64> = GRAD.iter().zip(&grad_b).map(|(&a, &b)| a * b).collect();
    let sbb: Vec<f64> = grad_b.iter().map(|&g| g * g).collect();
    let json = |v: &[f64]| serde_json::to_string(v).expect("json");

    println!("=== libxc_rs vs PySCF's C libxc, pointwise, OPEN SHELL ===");
    let mut worst = 0.0f64;
    for (code, gga) in [
        ("slater,", false),
        (",vwn", false),
        ("svwn", false),
        ("pbe,", true),
        (",pbe", true),
        ("pbe,pbe", true),
        ("blyp", true),
        ("b3lyp", true),
        ("pbe0", true),
    ] {
        let spec = libxc::parse_xc(code).unwrap_or_else(|e| panic!("{code} parses: {e:?}"));
        let (sa, sb, sc) = if gga {
            (Some(&saa[..]), Some(&sab[..]), Some(&sbb[..]))
        } else {
            (None, None, None)
        };
        let got = XcBackend::Libxc
            .eval_uks(&spec, &RHO, &rho_b, sa, sb, sc, DerivOrder::Vxc)
            .unwrap_or_else(|e| panic!("{code} eval_uks: {e:?}"));
        let up = run_script(
            &py,
            ORACLE_UKS_PY,
            &[
                code.to_string(),
                json(&RHO),
                json(&rho_b),
                json(&GRAD),
                json(&grad_b),
                if gga { "gga".into() } else { "lda".into() },
            ],
        );
        println!(
            "  {code}  ({} libxc component(s), {} points)",
            spec.components().len(),
            RHO.len()
        );
        let mut channels = vec![
            ("exc", &got.exc, "exc"),
            ("vrho_a", &got.vrho_a, "vrho_a"),
            ("vrho_b", &got.vrho_b, "vrho_b"),
        ];
        if gga {
            channels.push(("vs_aa", &got.vsigma_aa, "vsigma_aa"));
            channels.push(("vs_ab", &got.vsigma_ab, "vsigma_ab"));
            channels.push(("vs_bb", &got.vsigma_bb, "vsigma_bb"));
        }
        for (name, g, key) in channels {
            let w = pull(&up, key).unwrap_or_else(|| panic!("upstream {key}"));
            let c = compare(g, &w);
            report(name, &c);
            worst = worst.max(c.worst_ulp);
        }
    }
    println!("OPEN-SHELL worst deviation: {worst:.3} ULP");
    assert!(
        worst == 0.0,
        "open-shell libxc_rs is not bit-identical to PySCF's C libxc: worst {worst:.3} ULP \
         (see the log above for the channel and point)"
    );
}

//! `svml_pow8` vs numpy's `np.power(x, e)` bits — the F6 SVML validation.
//!
//! Gated on `PYSCF_ORACLE_VENV` like the rest of the upstream layer. Random
//! `x` values are generated in Rust, shipped as raw `u64` bits, and numpy
//! evaluates `x**e` on the AVX-512 host; the returned bits are compared with
//! `to_bits()`. Every `e` in `{l + 1.5}` and the whole `_env` exponent range
//! (`2·α` and `α_i + α_j` for α ∈ [0.1, 1e4]) must match with 0 mismatches.

use pyscf_gto::svml_pow::svml_pow8;
use std::path::PathBuf;
use std::process::Command;

const GATE: &str = "PYSCF_ORACLE_VENV";

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn oracle_python() -> Option<PathBuf> {
    let raw = std::env::var(GATE).ok()?.trim().to_string();
    if raw.is_empty() {
        return None;
    }
    let p = if matches!(raw.as_str(), "1" | "true" | "auto" | "yes") {
        workspace_root().join(".venv/bin/python")
    } else {
        let c = PathBuf::from(&raw);
        if c.is_dir() { c.join("bin/python") } else { c }
    };
    assert!(p.exists(), "{GATE} = {raw:?} -> {p:?}");
    Some(p)
}

const ORACLE_PY: &str = r#"
import os
os.environ['OMP_NUM_THREADS'] = '1'
import json, struct, sys
import numpy as np

with open(sys.argv[1]) as f:
    x_bits = json.load(f)
xs = np.array([struct.unpack('<d', struct.pack('<Q', b))[0] for b in x_bits])
out = {}
for e in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]:
    p = np.power(xs, e)
    out[str(e)] = [int(v) for v in np.ascontiguousarray(p).view(np.uint64)]
print(json.dumps(out))
"#;

fn run_python(py: &PathBuf, script: &str, bits: &[u64]) -> serde_json::Value {
    let dir = std::env::temp_dir();
    let base = format!("svml_pow_oracle_{}", std::process::id());
    let path = dir.join(format!("{base}.py"));
    let data = dir.join(format!("{base}.json"));
    std::fs::write(&path, script).expect("write oracle script");
    std::fs::write(&data, serde_json::to_string(bits).expect("json")).expect("write bits");
    let root = workspace_root();
    let out = Command::new(py)
        .arg(&path)
        .arg(&data)
        .env("PYTHONPATH", &root)
        .current_dir(&root)
        .output()
        .expect("spawn upstream python");
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&data);
    assert!(
        out.status.success(),
        "oracle failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .unwrap_or_else(|| panic!("oracle produced no JSON:\n{stdout}"))
        .to_string();
    serde_json::from_str(&line).expect("oracle JSON parses")
}

/// A deterministic split-mix PRNG (no external deps in the test).
fn rng_state() -> u64 {
    0x9E3779B97F4A7C15u64 ^ (std::process::id() as u64).rotate_left(17)
}
fn next(r: &mut u64) -> u64 {
    *r = r.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *r;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}
fn rand_double(r: &mut u64) -> f64 {
    (next(r) >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
}

#[test]
#[ignore = "F6: needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn svml_pow8_matches_numpy_power_bits() {
    let Some(py) = oracle_python() else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };

    // x values: log-uniform over the `_env` exponent range, plus the
    // dangerous mantissas (powers of two, and values near the 1/32 reciprocal
    // grid where `round5(vrcp14) != round5(1/x)`).
    let mut r = rng_state();
    let mut xs: Vec<f64> = Vec::new();
    let mut add = |v: f64| {
        if v.is_finite() && v > 0.0 {
            xs.push(v)
        }
    };
    for _ in 0..200_000 {
        let e = -3.0 + 10.0 * rand_double(&mut r);
        add(10.0_f64.powf(e));
    }
    for k in 0..32 {
        let base = 1.0 + k as f64 / 32.0;
        for &d in &[-1e-6, 0.0, 1e-6] {
            add(base + d);
        }
        add(2.0 * base);
        add(0.5 * base);
    }
    for k in -12..12 {
        add(2.0_f64.powi(k));
        add(1.5 * 2.0_f64.powi(k));
    }
    let x_bits: Vec<u64> = xs.iter().map(|x| x.to_bits()).collect();
    let want = run_python(&py, ORACLE_PY, &x_bits);

    let mut total = 0usize;
    let mut bad = 0usize;
    let mut per_e: std::collections::HashMap<String, usize> = Default::default();
    let mut plus = 0usize;
    let mut minus = 0usize;
    for e in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5] {
        let key = format!("{e}");
        let want_bits: Vec<u64> = want[&key]
            .as_array()
            .expect("bits array")
            .iter()
            .map(|v| v.as_u64().expect("u64"))
            .collect();
        assert_eq!(want_bits.len(), xs.len());
        for (i, &x) in xs.iter().enumerate() {
            total += 1;
            let got = svml_pow8(x, e);
            if got.to_bits() != want_bits[i] {
                bad += 1;
                *per_e.entry(key.clone()).or_insert(0) += 1;
                if got.to_bits() == want_bits[i].wrapping_add(1) {
                    plus += 1;
                } else if got.to_bits() == want_bits[i].wrapping_sub(1) {
                    minus += 1;
                }
                if bad <= 10 {
                    eprintln!(
                        "e={e} x_bits={:016x} (x={x}): got {:#x} ({got}) want {:#x} ({})",
                        x.to_bits(),
                        got.to_bits(),
                        want_bits[i],
                        f64::from_bits(want_bits[i])
                    );
                }
            }
        }
    }
    println!("per-e failures: {per_e:?}   +1ulp={plus}  -1ulp={minus}");
    assert_eq!(
        bad, 0,
        "svml_pow8 differs from numpy's np.power in {bad}/{total} cases"
    );
    println!("svml_pow8 matches numpy on {total} (x, e) pairs, 0 mismatches");
}
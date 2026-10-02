# T02 — Level shift: red test

**Goal.** A test that fails today and states upstream's formula.

**Upstream formula** (`pyscf/scf/hf.py:794-795`):

```
level_shift(s, d, f, factor) = f + (s − s·d·s) · factor
```

There is no `½` in it. Today `crates/pyscf-pbc-scf/src/kscf.rs:54` computes
`f + factor·(s − ½·s·d·s)`.

## Read first

- `crates/pyscf-pbc-scf/src/kscf.rs` lines 36–85 (`damp`, `level_shift`, `mm`).
- `crates/pyscf-pbc-scf/tests/grad_fc_hoist.rs` (shape of a test in this crate).

## Do

1. In `crates/pyscf-pbc-scf/src/kscf.rs` change `fn level_shift(` to
   `pub fn level_shift(`. Change nothing else in this task.
2. Create `crates/pyscf-pbc-scf/tests/level_shift_formula.rs`:

```rust
//! `kscf::level_shift` is upstream's `mol_hf.level_shift`
//! (`scf/hf.py:794-795`): `F' = F + (S − S·D·S)·factor`, with no factor ½.

use pyscf_algebra::CTensor;
use pyscf_pbc_scf::kscf::level_shift;

/// Naive complex `n × n` product, row-major.
fn mm(a: &CTensor, b: &CTensor, n: usize) -> CTensor {
    let mut out = CTensor::zeros(n * n);
    for i in 0..n {
        for j in 0..n {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for t in 0..n {
                let (ar, ai) = (a.re[i * n + t], a.im[i * n + t]);
                let (br, bi) = (b.re[t * n + j], b.im[t * n + j]);
                re += ar * br - ai * bi;
                im += ar * bi + ai * br;
            }
            out.re[i * n + j] = re;
            out.im[i * n + j] = im;
        }
    }
    out
}

#[test]
fn identity_overlap_shifts_occupied_down_and_virtual_up() {
    // S = 1, D = diag(2, 0, 0) (a restricted density), F = 0, factor 0.3
    // upstream: F' = 0.3·(1 − D) = diag(−0.3, +0.3, +0.3)
    let n = 3;
    let mut s = CTensor::zeros(n * n);
    for i in 0..n {
        s.re[i * n + i] = 1.0;
    }
    let mut d = CTensor::zeros(n * n);
    d.re[0] = 2.0;
    let mut fock = vec![vec![CTensor::zeros(n * n)]];
    level_shift(&mut fock, &vec![s], &vec![vec![d]], 0.3, n);
    let want = [-0.3, 0.3, 0.3];
    for i in 0..n {
        let got = fock[0][0].re[i * n + i];
        assert!((got - want[i]).abs() < 1e-14, "diagonal {i}: got {got}, want {}", want[i]);
    }
}

#[test]
fn general_matrices_match_the_formula() {
    let n = 4;
    // Any Hermitian S and D will do: the formula is plain linear algebra.
    let mut s = CTensor::zeros(n * n);
    let mut d = CTensor::zeros(n * n);
    let mut f = CTensor::zeros(n * n);
    for i in 0..n {
        for j in 0..n {
            let (a, b) = (i.min(j) as f64, i.max(j) as f64);
            let sign = if i <= j { 1.0 } else { -1.0 };
            s.re[i * n + j] = if i == j { 1.0 } else { 0.1 / (1.0 + a + b) };
            s.im[i * n + j] = if i == j { 0.0 } else { sign * 0.03 * (b - a) };
            d.re[i * n + j] = 0.5 / (1.0 + a + 2.0 * b);
            d.im[i * n + j] = if i == j { 0.0 } else { sign * 0.02 * (a + 1.0) };
            f.re[i * n + j] = 0.2 * (a - b);
            f.im[i * n + j] = if i == j { 0.0 } else { sign * 0.01 };
        }
    }
    let sds = mm(&mm(&s, &d, n), &s, n);
    let factor = 0.25;
    let mut fock = vec![vec![f.clone()]];
    level_shift(&mut fock, &vec![s.clone()], &vec![vec![d]], factor, n);
    for i in 0..n * n {
        let wr = f.re[i] + factor * (s.re[i] - sds.re[i]);
        let wi = f.im[i] + factor * (s.im[i] - sds.im[i]);
        assert!((fock[0][0].re[i] - wr).abs() < 1e-13, "re[{i}]");
        assert!((fock[0][0].im[i] - wi).abs() < 1e-13, "im[{i}]");
    }
}
```

## Verify

TEST form: `-p pyscf-pbc-scf --test level_shift_formula`.
Expected NOW: **both tests FAIL** (the first with `diagonal 0: got 0, want -0.3`).
A failing run is the correct result of this task.

## If it fails (meaning: it does not compile)

- `level_shift` not found: check step 1 and that `kscf` is `pub mod` in
  `crates/pyscf-pbc-scf/src/lib.rs`.
- `CTensor::zeros` / field names: open `crates/pyscf-algebra/src/complex.rs`
  and use the constructors defined there.

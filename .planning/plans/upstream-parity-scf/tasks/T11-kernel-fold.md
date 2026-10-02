# T11 — Kernel K-PP3: the fold `B += S · A`

**Goal.** A CubeCL kernel that accumulates
`B[row, q] += Σ_{g < nb} S[row, g] · A[q, g]` (complex), and a test showing
that splitting `g` into blocks changes no bit.

## Read first (required before writing the kernel)

CubeCL manual, directory
`/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/`:
- `Cubecl_generics.md` — whole file (generic `F: Float` kernels, launch order
  `::<F, R>`).
- `Cubecl_loop_control.md` — "continue (Not Supported Yet)".
- `11_launch_overhead_and_transfers.md` — §3 and §4.

Repo templates (copy their shape):
- `crates/pyscf-kernels/src/pbc/gv.rs` lines 30–150 — a generic `F: Float`
  kernel with its `launch_*_on_handles`, host launcher and public entry.
- `crates/pyscf-kernels/src/pbc/band_vmat.rs` lines 63–148 — a lane that
  starts from the value already in the output (`accumulate`).
- `crates/pyscf-kernels/src/pbc/ft_aopair.rs` lines 218–232 — `launch_1d`.

## Layout (all planar, `f64` on the host)

| buffer | shape | index |
|---|---|---|
| `s_re`, `s_im` | `(nrow, nbs)` | `row * nbs + g` |
| `a_re`, `a_im` | `(nao, nbs)` — AO-major | `q * nbs + g` |
| `b_re`, `b_im` | `(nrow, nao)` | `row * nao + q` |

`nbs` is the allocated block length, `nb ≤ nbs` the number of valid points.

## Do

1. Create `crates/pyscf-kernels/src/pbc/pp_gspace.rs`; add
   `pub mod pp_gspace;` to `crates/pyscf-kernels/src/pbc/mod.rs`.
2. Header and imports: copy lines 1–32 of `gv.rs` and adapt the text to
   "K-PP1..3 — the reciprocal-space non-local pseudopotential
   (`pyscf/pbc/df/fft.py:114-176`)". Also import
   `use pyscf_algebra::launch::{launch_1d, upload};`.
3. The kernel:

```rust
/// K-PP3 — `B[row, q] += Σ_{g < nb} S[row, g] · A[q, g]`. One lane owns one
/// `(row, q)` element and walks `g` upward, starting from the value already
/// in `B`. One accumulator per element and a fixed `g` order make the result
/// independent of how `g` was split into blocks, bit for bit.
///
/// Generic over the device float (`F: Float`, AGENTS.md §3).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn pp_fold_kernel<F: Float>(
    s_re: &Array<F>,
    s_im: &Array<F>,
    a_re: &Array<F>,
    a_im: &Array<F>,
    b_re: &mut Array<F>,
    b_im: &mut Array<F>,
    nrow: usize,
    nao: usize,
    nbs: usize,
    nb: usize,
) {
    let i = ABSOLUTE_POS;
    if i < nrow * nao {
        let row = i / nao;
        let q = i % nao;
        let sb = row * nbs;
        let ab = q * nbs;
        let mut sr = b_re[i];
        let mut si = b_im[i];
        for g in 0..nb {
            let xr = s_re[sb + g];
            let xi = s_im[sb + g];
            let yr = a_re[ab + g];
            let yi = a_im[ab + g];
            sr += xr * yr - xi * yi;
            si += xr * yi + xi * yr;
        }
        b_re[i] = sr;
        b_im[i] = si;
    }
}
```

4. `launch_pp_fold_on_handles<R: Runtime, F: DeviceScalar>(client, s_re,
   s_im, a_re, a_im, b_re, b_im: &Handle, nrow, nao, nbs, nb)`:
   - `let (count, dim) = launch_1d(client, nrow * nao, nb);`
   - `unsafe { pp_fold_kernel::launch_unchecked::<F, R>(client, count, dim,
     ArrayArg::from_raw_parts(...) × 6, nrow, nao, nbs, nb); }`
   - element counts for `from_raw_parts`: `nrow*nbs`, `nrow*nbs`, `nao*nbs`,
     `nao*nbs`, `nrow*nao`, `nrow*nao`;
   - a `// SAFETY:` comment stating those lengths and the `i < nrow*nao`
     guard.
5. Public test entry (host slices in, host vectors out), modelled on
   `launch_gv` + `gv` in `gv.rs`:

```rust
/// K-PP3 on host slices: returns the updated `(b_re, b_im)`.
///
/// # Errors
/// [`AlgebraError::ShapeMismatch`] when a slice length does not match
/// `nrow`, `nao`, `nbs`, or when `nb > nbs`.
#[allow(clippy::too_many_arguments)]
pub fn pp_fold(
    client: &AlgebraClient,
    s_re: &[f64], s_im: &[f64],
    a_re: &[f64], a_im: &[f64],
    b_re: &[f64], b_im: &[f64],
    nrow: usize, nao: usize, nbs: usize, nb: usize,
) -> Result<(Vec<f64>, Vec<f64>), AlgebraError>
```
   Validate every length first; upload the six slices with
   `upload::<R, f64>(client, slice)`; launch; read back
   `client.read(vec![b_re_h, b_im_h])` in ONE call; convert with
   `bytemuck::cast_slice::<u8, f64>`. Use `dispatch_backend!(client, c, Rt,
   …)` exactly as `gv.rs` does.
6. Create `crates/pyscf-kernels/tests/pbc_pp_fold.rs` (copy the `Lcg` and
   `cpu_client` helpers from `tests/pbc_gv.rs` lines 20–40):
   - `fold_matches_the_host_sum`: `nrow = 5`, `nao = 7`, `nbs = nb = 33`,
     random planes, `B` starts random. Host reference = the same double loop
     in plain Rust. Assert every element within `1e-12`.
   - `fold_is_bit_identical_across_blocks`: same data. (a) one call with all
     33 points. (b) three calls with `nbs = 13`: points `0..13`, `13..26`,
     `26..33` (copy each block into fresh `(nrow, 13)` / `(nao, 13)` buffers;
     the last call has `nb = 7`), feeding the returned `B` into the next
     call. Assert `a.to_bits() == b.to_bits()` for every element of both
     planes.

## Verify

1. CHECK `-p pyscf-kernels` → compiles with no error.
2. TEST `-p pyscf-kernels --test pbc_pp_fold` → `2 passed`.

## If it fails

- Compile error mentioning `cubecl`, `Float`, `launch_unchecked`, or
  `expand`: open the error guide directory named in `README.md` rule 5
  first.
- `mismatched types` on `launch_unchecked::<F, R>`: generic order is the
  kernel's own generics first, then the runtime (`Cubecl_generics.md`,
  "Launching a Generic Kernel").
- The bit-identity test fails: check that the lane reads `b_re[i]` BEFORE
  the loop and that `g` runs upward. Do not replace `to_bits` by a
  tolerance.
- The CPU runtime aborts (SIGSEGV / stack overflow): you added a local
  array or changed the launch size. Remove it; use `launch_1d`.

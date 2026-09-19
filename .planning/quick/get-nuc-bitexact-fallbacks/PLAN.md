# PLAN — Remove every bit-exactness fallback in `FFTDF.get_nuc`

Audience: an executor LLM. Follow the tasks **in order**. Do not skip the
"Verify" step of a task. Do not start a task before the previous one's
acceptance test passes.

---

## 0. Read this first (context you must not re-derive)

### 0.1 Where we are

`crates/pyscf-pbc-df/src/fftdf.rs::get_nuc` reproduces upstream PySCF 2.12.1
**bit for bit** on He/STO-3G, 2x2x2 k-points, mesh 11 — proven by
`crates/pyscf-pbc-df/tests/get_nuc_bitexact.rs`. That test currently PINS
`cell.mol._env` to upstream's values (see Task 6).

The exact pipeline is:

| Stage | Rust code | Upstream it copies |
|---|---|---|
| grid coords | `pyscf_pbc_gto::gv::fftfreq` (`i * (1/n)`) | `np.fft.fftfreq` |
| `absG2` / `coulG` | `pyscf_pbc_tools::coulg::abs_g2` (`(x²+z²)+y²`) | `np.einsum('gi,gi->g')` |
| `vneR` | `pyscf_pbc_tools::ifft_upstream` → `pocketfft.rs` | `scipy.fft.ifftn` |
| AO table | `pyscf_pbc_gto::eval_ao_kpts_upstream` (`eval_gto_upstream.rs`) | `libpbc` `PBCeval_sph_iter` |
| contraction | `pyscf_algebra::openblas_emu::{dgemm_nt,zgemm_nt}` | `lib.dot` → OpenBLAS 0.3.3 Barcelona |

### 0.2 The fallbacks you must remove

Each one currently makes `get_nuc` silently use a NON-exact path:

| # | Fallback | Where it falls back | Task |
|---|---|---|---|
| F1 | grid larger than one `aoR_loop` block | contraction assumes one block | Task 1 |
| F2 | basis with a shell `l >= 2` (or `cell.cart`) | `eval_ao_kpts_upstream` returns `None` | Task 2 |
| F3 | FFT axis that pocketfft plans with Bluestein (89, 101, 103, …) | `c2c_3d` returns `None` | Task 3 |
| F4 | mesh with ALL three axes in `_EXCLUDE` (e.g. 17, 47) | `ifft_upstream` returns `None` | Task 4 |
| F5 | more than one atom, or an atom off the origin | `rhoG` / `SI` not modelled | Task 5 |
| F6 | `cell._env` coefficients 1 ulp off upstream | test pins upstream `_env` | Task 6 |

### 0.3 Hard rules

1. **No FFI.** Production Rust must not link, `dlopen`, or call any C
   library (no `libopenblas`, no `libcint`, no `libm` beyond what `std`
   already uses, no `extern "C"`). Everything is re-implemented in Rust.
   *Python-side* `ctypes` inside oracle/probe scripts is allowed — that is
   the reference, not the product.
2. **Tests live in separate files** (`AGENTS.md` §2). Never add `mod tests`
   to a `src/` file.
3. **No cubecl kernels** in this plan. All new code is host Rust. (If you
   ever touch a cubecl kernel, `AGENTS.md` §3 requires reading the CubeCL
   manual first.)
4. **Format only files you touched:** `rustfmt --edition 2024 <files>`.
   Never run bare `cargo fmt` (it reformats unrelated files).
5. **Never trust a formula you did not measure.** Every rounding order in
   this plan was found by *probing upstream and comparing bits*. When a task
   says "probe", write a Python script that prints bit patterns and compare;
   do not guess from reading source alone.
6. Put probe scripts, downloaded sources and logs in the session scratchpad
   or `target/`, **never** in the repo and never in `/tmp` (it is a RAM
   tmpfs).
7. Do not commit. Report at the end.

### 0.4 Oracle facts (verified 2026-09-18 — reuse, don't rediscover)

- Upstream = vendored `./pyscf` (2.12.1) run by `.venv/bin/python` with
  `PYTHONPATH=<repo root>`. Its compiled `.so` files come from
  `.venv/lib/python3.13/site-packages/pyscf/lib/`.
- Upstream C code calls **OpenBLAS 0.3.3**, core **Barcelona** (SSE2, **no
  FMA**). numpy uses a **different** BLAS (its own, with FMA) — do not mix
  them up.
- `lib.dot` (`NPdgemm`/`NPzgemm`) splits K over OpenMP threads and merges in
  `omp critical` order → **every oracle must start with**
  `import os; os.environ['OMP_NUM_THREADS']='1'` **before** importing numpy.
- Ship floats as raw bits: `[int(v) for v in arr.ravel().view(np.uint64)]`;
  compare with `f64::to_bits()` in Rust. Never compare decimals.
- numpy 1.26 on this AVX-512 host evaluates array `x**y` with **SVML**, not
  glibc `pow` (differs in ~28% of cases). Scalar Python `math.pow` is glibc.
- `scipy.special.gamma(1.5)` (cephes) = `0x1.c5bf891b4ef6ap-1`, 1 ulp below
  `math.gamma(1.5)`.
- Rust `f64::exp/ln/cos/sin/sqrt` call glibc on Linux → match upstream C's
  `exp`, `log`, `cos`, `sin`, `sqrt`.
- Rust never contracts `a*b+c` into FMA. Use `f64::mul_add` **only** where a
  probe proved upstream fuses.

### 0.5 Commands you will reuse

```bash
# the bit-exact gate (+ BLAS emulator check)
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-df --release --test get_nuc_bitexact -- --ignored --nocapture
# oracle-free unit tests you will extend
cargo test --release -p pyscf-algebra --test openblas_emu
cargo test --release -p pyscf-pbc-tools --test pocketfft
# regression sweep before you finish (separate commands: `--test X` filters ALL -p packages)
cargo test --release --no-fail-fast -p pyscf-pbc-tools -p pyscf-pbc-gto
PYSCF_ORACLE_VENV=1 cargo test --release --no-fail-fast -p pyscf-pbc-df --test fftdf --test ft_ao --test get_nuc_bitexact -- --include-ignored
# lint (only look at findings in files you touched)
cargo clippy -p pyscf-algebra -p pyscf-pbc-tools -p pyscf-pbc-gto -p pyscf-pbc-df --all-targets
```

A Python probe runs as:

```bash
PYTHONPATH=. .venv/bin/python <scratch>/probe.py 2>&1 | grep -v -i plugin | grep -v "^/home"
```

### 0.6 The probing method (use it in every task)

1. Write a Python script that computes the upstream value with upstream's
   own code (e.g. call the real function, or `ctypes` into the wheel's
   `.so`).
2. In the same script, compute 3–10 **candidate** re-implementations in
   plain Python scalars (`float`, `math.fma`, `math.exp`, …), each a
   different operation order.
3. Print, per candidate, how many elements differ **in the bits** from
   upstream, over **many random inputs and shapes**.
4. Keep only a candidate with **0 mismatches**. If none: widen the family
   (accumulator counts, remainder placement, combine order) or brute-force
   all summation trees for small sizes (that is how the `dgemm` 1×2 tail
   was found: enumerate binary trees over `C, p0..pK-1`, keep those that
   match every trial, read off the pattern, then generalise).
5. Only then port the winning candidate to Rust, and add an oracle test that
   compares the Rust output bits to upstream.

---

## Task 1 — F1: grids split into several `aoR_loop` blocks

### Facts
`FFTDF.aoR_loop` → `KNumInt.block_loop` (`pyscf/pbc/dft/numint.py:1058-1100`):

```python
blksize = int(max_memory*1e6/(comp*2*nao*16*BLKSIZE))      # BLKSIZE = 56
blksize = max(4, min(blksize, ngrids//BLKSIZE+1, 2400)) * BLKSIZE
```

`max_memory = max(2000, mydf.max_memory - lib.current_memory()[0])` (in
`aoR_loop`, `fft.py:311`). `comp = 1`. Then `fft.py:73-74` does, per block,
`vne[k] += lib.dot(ao.T.conj()*vneR[p0:p1], ao)` starting from `vne[k] = 0`.

`current_memory()` is the Python process RSS → upstream's block size is
**not deterministic**. Make it deterministic in the oracle by setting
`mydf.max_memory = 0` (then `max_memory == 2000` exactly).

### Steps
1. In `crates/pyscf-pbc-df/src/fftdf.rs`, add a function
   `fn aor_loop_blocks(ngrids: usize, nao: usize, nkpts: usize, max_memory_mb: f64) -> Vec<(usize, usize)>`
   implementing the two lines above (note `nkpts` is not in the formula —
   check the source line again before you write it; use exactly the
   variables upstream uses).
2. Change `contract_local_potential_upstream` to loop over those blocks:
   for each block call the emulated `dgemm_nt`/`zgemm_nt` on the block's
   columns only (K = block length), then accumulate
   `vne = vne + block_result` element-wise, starting from `0.0`.
   Each block's GEMM result goes through `NPdgemm`'s `c = 0; c += cpriv`
   first (`+ 0.0`), then `vne[k] += ...`.
3. `max_memory_mb`: use `max(2000.0, df.max_memory - 0.0)` is WRONG
   (Rust has no RSS). Use the constant 2000.0 and document that the oracle
   pins `mydf.max_memory = 0`. Keep the AO evaluation over the whole grid
   (upstream's per-block evaluation is identical point by point because the
   56-point `BLKSIZE` blocks stay aligned — `blksize` is a multiple of 56).
4. Also check `NPdgemm`'s branch `if ((k/m) > 3 && (k/n) > 3)`: when it is
   false upstream calls `dgemm_` directly with `beta = 0` (no `+ 0.0`
   step, but OpenBLAS zeroes `C` first, so the value is identical). Keep
   the `+ 0.0` normalisation; add a comment.

### Test (add to `crates/pyscf-pbc-df/tests/get_nuc_bitexact.rs`)
- New `#[test] #[ignore]` fn `get_nuc_bit_identical_multi_block`: He/STO-3G,
  gamma only, mesh `[41, 41, 43]` (72 283 points) … choose a mesh so that
  `ngrids//56+1 > blksize/56` at `max_memory = 2000`; compute that in the
  test and `assert!(nblocks >= 2)` BEFORE comparing (a test that silently
  has one block proves nothing).
- The oracle script must set `mydf.max_memory = 0` before `get_nuc`.
- Note: 41 and 43 are both in `_EXCLUDE` → if all three axes are in
  `_EXCLUDE` the FFT falls back (F4). Pick a mesh with at least one axis
  **not** in `_EXCLUDE` and no Bluestein axis, e.g. `[45, 45, 45]`.

### Acceptance
The new test and the existing He test pass; `bit_diff == 0` on all
elements.

---

## Task 2 — F2: shells with `l >= 2` (and `cell.cart`)

### Facts
`PBCeval_sph_iter` (`pyscf/lib/pbc/grid_ao.c:301-430`):
- `l <= 1`: `feval` writes straight into `pao` (already ported).
- `l >= 2`: `feval` writes Cartesian values into `cart_gto`, then for each
  `(comp, contraction)` calls `CINTc2s_ket_sph1(pao, pcart, bgrids,
  bgrids, l)`.
- `feval` = `GTOshell_eval_grid_cart` (`pyscf/lib/gto/deriv1.c:129-...`):
  explicit expressions for `l = 0, 1, 2, 3`, a generic `xpows/ypows/zpows`
  loop for larger `l`. **Read the whole function** and copy each
  expression's operand order exactly (e.g. `l=2`:
  `exps * gridx * gridx` is `(exps*gridx)*gridx`).
- `fac = CINTcommon_fac_sp(l)` is `1.0` for `l >= 2`.
- `CINTc2s_ket_sph1` lives in **libcint** (not in `./pyscf`). The wheel ships
  `.venv/.../pyscf/lib/deps/lib/libcint.so` (no source).
- `cell.cart = True` uses `PBCeval_cart_iter` (no c2s step).

### Steps
1. Find the libcint version: run
   `strings .venv/lib/python3.13/site-packages/pyscf/lib/deps/lib/libcint.so | grep -i -E "^[0-9]+\.[0-9]+\.[0-9]+$|libcint"`
   and check `pyscf/lib/CMakeLists.txt` for the pinned libcint git tag.
2. Download the matching source with `curl` into the scratchpad (e.g.
   `https://raw.githubusercontent.com/sunqm/libcint/<tag>/src/cart2sph.c`).
   This is reading source, not FFI.
3. In `cart2sph.c` find `CINTc2s_ket_sph1` and the coefficient table it
   uses (`g_c2s[l].cart2sph` or hand-written per-`l` functions). Copy the
   arithmetic **order** exactly: whether it zeroes then accumulates, skips
   zero coefficients, and the loop nesting.
4. Coefficient values: copy them as **decimal literals exactly as written
   in the C source** into a new file
   `crates/pyscf-pbc-gto/src/c2s_tables.rs`. Then **probe** (Step 6) that
   each literal parses to the same `f64` bits as libcint's compiled value —
   extract compiled values with a Python probe that calls
   `pyscf.gto.mole.cart2sph(l)` (it calls `CINTc2s_ket_sph` on an identity
   matrix, `mole.py:157-187`) and prints bits.
5. Extend `eval_gto_upstream.rs`:
   - port `GTOshell_eval_grid_cart` for every `l` up to the basis max
     (explicit branches `l = 2, 3`; generic branch above),
   - port `CINTc2s_ket_sph1`,
   - for `l >= 2` write Cartesian values to a scratch buffer and transform
     into `pao` exactly as `PBCeval_sph_iter` does (`pao += deg*bgrids`,
     `pcart += dcart*bgrids` per `(comp, contraction)`),
   - add a `cart` branch that mirrors `PBCeval_cart_iter` (read it:
     `grid_ao.c` near line 186) when `cell.mol.cart` is true,
   - remove the `if shells.iter().any(|s| s.l > 1) { return Ok(None) }`
     early return; return `None` only for `l` beyond what you ported
     (e.g. `l > 7`), and document it.
6. Probe **before** trusting the port: in Python, evaluate
   `numint.eval_ao_kpts(cell, coords, kpts)` for a cell with d/f shells and
   compare bits against a Python-scalar re-implementation of your port.
   Fix order issues in Python first, then in Rust.

### Tests
- `crates/pyscf-pbc-gto/tests/eval_gto_upstream.rs` (new, oracle-free):
  `l = 0..4` shells — Rust upstream-order values vs the production
  `eval_ao_kpts` agree to `1e-12` (catches wrong formulas, not bits).
- `get_nuc_bitexact.rs`: new ignored test `get_nuc_bit_identical_d_shells`
  on He with `cc-pvtz` (has d) — and add a stage assertion on the AO table.
  Keep pinning `_env` until Task 6 is done.
- Add a case with `cell.cart = true` (He `cc-pvtz`, cart).

### Acceptance
AO table and `get_nuc` bit-identical for He/cc-pVTZ, sph and cart.

---

## Task 3 — F3: Bluestein-planned FFT axes

### Facts
- `crates/pyscf-pbc-tools/src/pocketfft.rs` ports `cfftp` only.
  `uses_bluestein(n)` is already exact (first Bluestein length: 89).
- Source: scipy 1.17.1's pocketfft submodule, commit
  `9367142748fcc9696a1c9e5a99b76ed9897c9daa`, file `pocketfft_hdronly.h`,
  class `fftblue` (~line 2380–2480). Download:
  `curl -sSfL -o <scratch>/pocketfft.h https://raw.githubusercontent.com/scipy/pocketfft/9367142748fcc9696a1c9e5a99b76ed9897c9daa/pocketfft_hdronly.h`

### Steps
1. Read `fftblue`: constructor (`n2 = good_size_cmplx(n*2-1)`, `bk`, `bkf`
   tables built from `sincos_2pibyn<T0> tmp(2*n)` with the `coeff` index
   `m*m mod 2n` recurrence, `xn2 = T0(1)/T0(n2)` — note this one is
   **double**, not long double: check the type), and `fft<fwd>(c, fct)`
   (the `akf` multiply with `special_mul`, the `plan.exec` calls on the
   inner `cfftp` of length `n2`, the final `special_mul` and the `fct`
   application — copy exactly where `fct` is applied).
2. Port it as `struct Fftblue` in `pocketfft.rs`. Reuse `Cfftp`,
   `SinCos2PiByN`, `special_mul`, `good_size_cmplx` (already there).
3. Add `enum Plan1d { Pack(Cfftp), Blue(Fftblue) }` with `fn new(len)`
   mirroring `pocketfft_c` (the `uses_bluestein` decision), and use it in
   `c2c_3d`. Delete the `uses_bluestein` early `return None`.
4. Change `c2c_3d`'s return type from `Option<…>` to a plain value and
   update `ifft_upstream` accordingly (only F4 can still return `None`
   after this task).

### Tests
- `crates/pyscf-pbc-tools/tests/pocketfft.rs`: extend
  `cfftp_matches_a_naive_dft` to cover ALL `n` in `1..=200` through
  `Plan1d` (Bluestein included) at `1e-12` relative.
- New oracle test in `crates/pyscf-pbc-df/tests/get_nuc_bitexact.rs` (it has
  the Python harness): a standalone check that
  `ifft_upstream(random, mesh)` equals `scipy.fft.ifftn(…, axes=(1,2,3))`
  bits for meshes `[89, 12, 15]`, `[101, 9, 10]`, `[5, 6, 7]`. Generate the
  random input in Python, ship bits, compare bits.
- `get_nuc` on He with mesh `[89, 16, 16]` (not all `_EXCLUDE`).

### Acceptance
All three FFT meshes bit-identical; `get_nuc` on the 89-mesh bit-identical.

---

## Task 4 — F4: all-`_EXCLUDE` meshes (`_ifftn_blas`)

### Facts
`pyscf/pbc/tools/pbc.py:50-68` (`_ifftn_blas`):

```python
expRGx = np.exp(2j*np.pi*np.fft.fftfreq(mx)[:,None] * np.arange(mx))
...
blksize = max(int(1e5 / (mx * my * mz)), 8) * 4
for i0, i1 in lib.prange(0, n, blksize):
    f = lib.transpose(g[i0:i1].reshape(ni,-1), out=buf1.reshape(-1,ni))
    f = lib.dot(f.reshape(mx,-1).T, expRGx, 1./mx, c=out1.reshape(-1,mx))
    f = lib.dot(f.reshape(my,-1).T, expRGy, 1./my, c=buf1.reshape(-1,my))
    f = lib.dot(f.reshape(mz,-1).T, expRGz, 1./mz, c=out1.reshape(-1,mz))
```

Three unknowns you must probe:
1. **`expRG`** — numpy array arithmetic: `2j*np.pi` (complex scalar),
   `* fftfreq[:,None]` (complex × float broadcast), `* arange` (complex ×
   int), then `np.exp` on a complex **array** (may be SIMD, not `cexp`).
   Probe each operation separately against Python-scalar candidates.
   If `np.exp(complex array)` is not reproducible with `(exp(re)*cos(im),
   exp(re)*sin(im))`, find numpy 1.26's complex exp loop in the numpy
   source (download numpy v1.26.4 `numpy/core/src/umath/loops_umath_fp.dispatch.c.src`
   / `npy_math_complex.c.src`) and port it.
2. **The GEMM call shape.** `lib.dot(a, b, alpha, c)` with `a` an
   F-contiguous transposed view: work out (from `numpy_helper.py` `zdot`
   and `_zgemm`) the exact `zgemm_` arguments: `transa`, `transb`, `m, n,
   k`, `alpha = (1/mx, 0)`, `beta = 0`. Also check `NPzgemm`'s
   `(k/m) > 3 && (k/n) > 3` branch (here `k = mx` is small, so it is
   normally the direct call).
3. **The Barcelona `zgemm` model for that shape**, including `alpha != 1`.
   `openblas_emu::zgemm_nt` only covers `('N','T')` with `alpha = 1`.

### Steps
1. Probe `expRG` (unknown 1). Port the winner as
   `fn exp_rg(n: usize) -> Vec<(f64, f64)>` in a new file
   `crates/pyscf-pbc-tools/src/fft_blas_upstream.rs`.
2. Probe the `zgemm_` shape (unknown 2) by printing the arguments in a
   Python re-implementation of `zdot`/`_zgemm` for the real array shapes.
3. Probe the kernel (unknown 3) with the ctypes harness already written in
   `get_nuc_bitexact.rs::BLAS_PY` (copy it into a scratch script, change the
   `trans` flags and `alpha`). Fit, per register-tile class (`zgemm` tile is
   2×2: classes "full" and "tail-1" for rows and columns), which
   accumulation order holds, and how `alpha` is applied: candidates
   `C = alpha*acc` as `(ar*accr - ai*acci, ar*acci + ai*accr)` vs
   `(accr*ar - acci*ai, …)` vs scaling before the sum. Check K-blocking is
   still `GEMM_Q = 224` (it is irrelevant when `k < 224`, but assert it).
4. Add `pub fn zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, c,
   ldc)` (planar complex) to `crates/pyscf-algebra/src/openblas_emu.rs`,
   covering the shapes you verified; keep `zgemm_nt` as a thin wrapper.
   Panic with a clear message on any `(transa, transb)` you did not verify.
5. Port `_ifftn_blas` in `fft_blas_upstream.rs` (`lib.transpose` is an exact
   copy; reproduce the reshapes as index arithmetic; `n = 1` batch for
   `get_nuc`, but support `n > 1` with the `blksize` loop).
6. In `fft.rs::ifft_upstream`, route all-`_EXCLUDE` meshes to it; the return
   type becomes `Result<CTensor, _>` (no more `None`).

### Tests
- `crates/pyscf-algebra/tests/openblas_emu.rs`: integer-input exactness test
  for the new `zgemm` shapes (like `integer_inputs_give_the_exact_product`).
- `get_nuc_bitexact.rs::openblas_emulation_matches_the_wheel_blas`: add the
  new `(trans, alpha)` shapes; K up to 60; M up to 5000 (the real `m` here is
  `ngrids/mx`).
- Oracle test: `ifft_upstream` vs `pyscf.pbc.tools.ifft` bits for meshes
  `[17,17,17]`, `[47,47,47]`, `[17,19,23]`.
- `get_nuc` bit-identical on He/STO-3G at mesh `[17,17,17]`.

### Acceptance
All listed meshes bit-identical; `ifft_upstream` no longer returns `Option`.

---

## Task 5 — F5: several atoms / atoms off the origin

### Facts
Two numpy-side stages, run on **numpy's** BLAS (not the wheel's):
1. `SI = cell.get_SI(mesh=mesh)` (`cell.py:612-645`) — for `Gv=None` it is
   the **separable** branch (`get_SI(mesh=mesh)` passes no `Gv`!). Read
   `cell.py:619-640`: `rb = np.dot(coords, b.T)`, `SIx = np.exp(-1j*np.einsum('z,g->zg', rb[:,0], basex))`
   (or similar — read it), then products `SIx[:,:,None,None] * SIy[:,None,:,None] * SIz[:,None,None,:]`.
   Rust `get_si` in `get_nuc` is called with `Some(&gv)` — the **dense**
   branch, which is a different computation. First confirm which branch
   upstream `fft.py:63` uses (`cell.get_SI(mesh=mesh)` → no Gv → separable)
   and switch Rust to the same branch (`get_si(cell, None, Some(mesh), None)`).
   He at the origin hid this: every phase is exactly 1.
2. `rhoG = numpy.dot(charge, SI)` (`fft.py:64`): `charge` float64
   `(natm,)`, `SI` complex128 `(natm, ngrids)`. Probe which numpy code path
   runs (type promotion → `zgemv`? or `cblas_zdotu`? or numpy's own loop)
   and its summation order/FMA use.

### Steps
1. Probe `SI` for diamond all-electron (2 C atoms, one off origin) and for a
   random 3-atom cell: compare upstream `c.get_SI(mesh=mesh)` bits against
   candidates for `rb` (`np.dot` 3-term order, with/without FMA), the
   complex `np.exp` of a pure-imaginary array (see Task 4 unknown 1 — reuse
   what you learned), and the product order `(SIx*SIy)*SIz`.
   The current Rust separable branch (`gv.rs:346-395`) computes
   `x*y` first then `*z` — verify, do not assume.
2. Fix `crates/pyscf-pbc-gto/src/gv.rs::get_si`'s separable branch to the
   winning candidate (this also affects Ewald and other callers — run their
   tests).
3. Probe `rhoG` for `natm = 2, 3, 5`. numpy's BLAS reports its core via
   `numpy.show_config()` / `threadpoolctl` (install it in a scratch venv if
   missing — do not add it to the repo). If it is a dot/gemv with FMA,
   model it with `f64::mul_add` exactly as probed.
4. Port into `get_nuc` (`fftdf.rs`, the `rho_re/rho_im` loop).

### Tests
- Oracle stage test: `SI` bits (diamond all-electron, 3-atom random cell).
- Oracle stage test: `rhoG` bits.
- `get_nuc` bit-identical: diamond **all-electron** `sto-3g` (d-free), and a
  2-atom He2 cell with one atom off-origin. Keep `_env` pinned (Task 6).
- Re-run `pyscf-pbc-gto` Ewald tests (`cargo test --release -p pyscf-pbc-gto`).

### Acceptance
Multi-atom `get_nuc` bit-identical with `_env` pinned; no Ewald regression.

---

## Task 6 — F6: `cell._env` contraction normalisation

This is the hardest task. It touches **every** integral (it is `Mole`
construction), so gate it behind a check before switching it on.

### Facts
`pyscf/gto/mole.py:984-1027`:

```python
cs = numpy.einsum('pi,p->pi', cs, gto_norm(angl, es))      # gto_norm on an ARRAY
cs = _nomalize_contracted_ao(angl, es, cs)                  # ee = gaussian_int(l*2+2, es_i+es_j) ARRAY
# gaussian_int(n, alpha) = scipy.special.gamma(n1) / (2. * alpha**n1),  n1 = (n+1)/2
# s1 = 1/numpy.sqrt(numpy.einsum('pi,pq,qi->i', cs, ee, cs))
```

Verified: replicating this with **numpy arrays** reproduces upstream
`_env` exactly; with glibc `pow` it does not. Rust version:
`crates/pyscf-gto/src/make_env.rs::normalise_contractions` (uses
`libm::tgamma`, `powf`, and product order `(c_i*c_j)*S_ij`).

### Steps
1. **Gamma (easy).** `n1 = l + 1.5` only. Build a table
   `CEPHES_GAMMA_HALF[l]` for `l = 0..=15` by printing
   `scipy.special.gamma(l+1.5)` bits in Python; store as `f64::from_bits`
   constants in `make_env.rs`. Do not compute gamma in Rust.
2. **einsum orders.** Probe `numpy.einsum('pi,pq,qi->i', cs, ee, cs)` and
   `'pi,p->pi'` / `'pi,i->pi'` with *given* arrays (feed upstream's own
   `ee`, `cs`) against Python-scalar candidates: product order
   `(a*b)*c` vs `a*(b*c)`, loop order `p`-outer vs `q`-outer, sequential vs
   pairwise/SIMD accumulation. Use many random `nprim` (1..12) and `nctr`
   (1..4).
3. **SVML `pow` (hard).** numpy's `a**n1` on float64 arrays on this host is
   Intel SVML (AVX-512). Source: numpy v1.26.4's `numpy/core/src/umath/svml`
   submodule (repo `numpy/SVML`, file `linux/avx512/svml_z0_pow_d_la.s`).
   - Download it; read the algorithm (table lookups + polynomial; it is
     generated assembly — follow the data flow instruction by instruction).
   - Re-implement it as **scalar Rust**: every `vfmadd*` → `f64::mul_add`,
     every `vmul/vadd/vsub` → plain ops, `vgetexp/vgetmant/vrndscale` →
     explicit bit manipulation, table loads → `const` arrays copied as hex.
   - Probe first in Python (a scalar re-implementation) against
     `np.power(x_array, e)` bits over ≥1e6 random `x` in the ranges
     `2*alpha` and `alpha_i+alpha_j` take (1e-3 .. 1e7) and every `e` in
     `{l+1.5}`. 0 mismatches required.
   - Also probe: does numpy use SVML for **every** array length (including
     1–7 elements, the masked tail)? `gto_norm` arrays are tiny. If short
     arrays take a different path, handle it.
   - Also check when the special-value paths (`x` subnormal, exponent
     overflow) trigger; basis exponents never hit them, but `debug_assert!`
     that inputs are in the probed range.
4. Implement `normalise_contractions` with the table gamma, the SVML-order
   `pow`, and the probed einsum orders. Put the SVML port in its own file
   `crates/pyscf-gto/src/svml_pow.rs` with the source commit in the module
   docs.
5. This changes every molecular and periodic integral by ulps. Run the
   broad test suites (see §0.5 plus `cargo test --release -p pyscf-gto`) and
   report any gate that moves. Do NOT loosen tolerances to make tests pass —
   report instead.
6. When `_env` matches: delete the `_env` pin in `get_nuc_bitexact.rs` and
   turn its `println!` of the `_env` diff into an `assert_eq!(n_env, 0)`.

### Tests
- `crates/pyscf-gto/tests/make_env_bitexact.rs` (new, oracle-gated): `_env`
  bits vs upstream for He/Ne/C/O/Fe × `sto-3g`, `6-31g`, `cc-pvdz`,
  `cc-pvtz`, `def2-svp`.
- `crates/pyscf-gto/tests/svml_pow.rs` (new, oracle-gated): 1e5 random
  inputs vs `np.power` bits, generated in Python and shipped as bits.

### Acceptance
`_env` bit-identical for every listed basis; `get_nuc_bitexact.rs` passes
with **no pinning**.

---

## Task 7 — Final integration

1. `get_nuc` must have **no** remaining fallback branch. Grep `fftdf.rs` for
   `None =>` / `ao_kpts(` inside `get_nuc`; the only allowed fallback is a
   documented `l > <max ported>` refusal (return an error, do not silently
   use the non-exact path — decide with the user if unsure).
2. Update the doc comment of `get_nuc` (it lists the fallbacks) and the module
   docs of `eval_gto_upstream.rs`, `pocketfft.rs`, `openblas_emu.rs`.
3. Gate matrix in `get_nuc_bitexact.rs` (all `#[ignore]`, oracle-gated),
   each asserting bits on every stage AND the result:

   | Case | Exercises |
   |---|---|
   | He/STO-3G, 2x2x2, mesh 11 | baseline |
   | He/STO-3G, gamma, mesh 45³ | F1 multi-block |
   | He/cc-pVTZ sph + cart, 2x2x2, mesh 15 | F2 |
   | He/STO-3G, mesh [89,16,16] | F3 |
   | He/STO-3G, mesh 17³ | F4 |
   | diamond all-electron STO-3G, 2x2x2, mesh 21 | F5 |
   | all of the above with no `_env` pin | F6 |

4. Run the full sweep in §0.5; all green. Run clippy; fix findings **in
   files you touched** only.
5. Write a short report: what each task changed, the probe that proved each
   order, any case still not exact and why.

---

## Traps that already cost time (read before debugging)

- `cargo test -p A --test X -p B` filters **all** packages to test `X`;
  B's tests silently do not run. Use separate commands.
- A stage can look exact on He at the origin because every phase is 1.
  Always include an off-origin atom and a non-gamma k-point.
- A random-input probe that passes on one shape proves nothing about
  another: register-tile tails (M or N not a multiple of 4 for `dgemm`, 2
  for `zgemm`) and K-blocks above 224 each have their own order.
- Truncated probe output (`[:15]`) once made a correct Rust result look
  wrong. Print full lists or counts.
- Upstream multi-threaded results differ from run to run. If an oracle
  value changes between two runs, you forgot `OMP_NUM_THREADS=1`.
- JSON floats may round-trip wrong. Ship `u64` bits.
- `rustfmt` hooks may rewrite files between your edits; re-read a file
  before editing it again.

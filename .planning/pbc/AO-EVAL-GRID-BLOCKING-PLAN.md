# AO-evaluation grid blocking (B-00 … B-04)

**Created:** 2026-09-21 · **Revised:** 2026-09-21 (rewritten for literal execution)
**Status:** draft — no code written.
**Follows from:** [`../phases/11-fft-fftdf-periodic-hf/measurements/k14f-hcore-fused-contraction.md`](../phases/11-fft-fftdf-periodic-hf/measurements/k14f-hcore-fused-contraction.md)

---

## 0. READ THIS FIRST

You are executing this document literally. **Do not infer. Do not improvise. Do not
substitute a similar approach.** If an instruction cannot be followed exactly as
written, go to §0.5 STOP CONDITIONS.

Sections marked **WHY** are background. You do **not** need to read them to execute.
Sections marked **DO** are instructions. Execute them in the order given.

### 0.1 Vocabulary — every symbol used below

| symbol | meaning | where it comes from |
|---|---|---|
| `nkpts` | number of k-points | `kpts.len()` |
| `nao` | number of atomic orbitals | `cell.mol.nao_nr` |
| `ngrids` | number of grid points | `mesh[0] * mesh[1] * mesh[2]` |
| `comp` | components per AO | `1` for `GTOval_sph`, `4` for `GTOval_sph_deriv1` |
| `n` | reals per k-point in the accumulator | `comp * ngrids * nao` |
| `BLK` | grid points per block (introduced by B-03) | new, tunable |
| "the table" | the complex AO table | `16 * nkpts * nao * ngrids` bytes |
| "the accumulator" | `AoKAccumulator`'s two device f64 planes | `2 * nkpts * n * 8` bytes |
| "peak Δ" | peak RSS minus RSS before the call | `VmHWM` − `VmRSS`, see C4 |

### 0.2 Hard rules — each one is checkable, none is advisory

| # | rule | how it is checked |
|---|---|---|
| R1 | No `mod tests` inside a production source file. Tests go in `crates/<crate>/tests/<name>.rs`. | `grep -n "mod tests" <file>` must find nothing new |
| R2 | Before editing any `#[cube]` kernel, read `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md`. | — |
| R3 | On ANY cubecl build error: STOP. Read `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/`. Do not attempt a fix before reading it. | — |
| R4 | Only `pyscf-algebra`, `pyscf-runtime`, `pyscf-kernels` may name a `cubecl-*` type. New kernels go in `crates/pyscf-kernels/src/pbc/`. | command **C3** |
| R5 | Every kernel is generic over the device float: `fn k<F: Float>(...)`. Write `F::from_int(0)`, never `F::new(0.0)` — the latter fails clippy under `-D warnings`. | command **C2** |
| R6 | Change ONE item, then run the full §0.4 command set, then record the result. Never stack two items before measuring. | — |
| R7 | A peak-RSS comparison is **one call per process**. Never two arms in one process. Never more than one rep before reading the peak. | §0.3 trap T1 |
| R8 | Every rep gets a **fresh `Fftdf`**. Never reuse one across reps. | §0.3 trap T2 |
| R9 | `rustfmt` only the files you touched. Never run a bare `cargo fmt`. | command **C6** |

### 0.3 Traps — these already cost one session; do not rediscover them

| # | trap | consequence if ignored |
|---|---|---|
| T1 | CubeCL pools device buffers per **process-global** client. Arm 2 in the same process inherits arm 1's pool. | Two arms that really differed by 260 MiB both reported 2680 MiB. |
| T2 | A shared `Fftdf` lets rep 2+ hit the AO cache. | The arm reports a cached 31 ms reduction as if it were the whole 2078 ms call. |
| T3 | `ulimit -v` makes **every** CubeCL launch panic. | False gate failures. Use `/proc/self/clear_refs` (see C4) to reset the watermark, never a virtual-size cap. |
| T4 | `client.empty()` returns a **dirty recycled** buffer. | Garbage read as zeros. Always fill before first read. |
| T5 | A process-global env switch races across concurrent test bodies. | `tests/hcore_fused.rs` failed at `--test-threads=4` and passed at 2. Reuse its `FUSE_SWITCH` mutex pattern. |
| T6 | `cargo test --release` rebuilds the dependency tree with `panic=unwind`, and libxc is ~500 crates. | First run can take **over an hour**. It is not hung. Run it detached with output to a log. |

### 0.4 Commands — run these verbatim, from the repository root

```bash
# C1  full test suite for the three crates this plan touches
#     EXPECTED (baseline at commit d985e49, 2026-09-21):
#       450 passed; 0 failed; 63 ignored
cargo test --release -p pyscf-kernels -p pyscf-pbc-gto -p pyscf-pbc-df -- --test-threads=4

# C2  lint — CI runs this with -D warnings, so a warning is a failure
cargo clippy -p pyscf-kernels -p pyscf-pbc-gto -p pyscf-pbc-df --tests -- -D warnings

# C3  algebra wall (R4)
cargo run -p xtask --bin check-dependency-wall

# C4  peak-RSS probe — ONE call per process (R7). Run the loop exactly as written.
for t in ao_table_only host_route fused_route; do
  PROBE_NK=3 PROBE_MESH=41 cargo test --release -p pyscf-pbc-df \
    --test hcore_peak_rss -- --ignored --nocapture --exact "$t"
done

# C5  time A/B — build once, then one process per arm
cargo build --release -p pyscf-bench --bin krks_profile
for arm in host fused; do
  ./target/release/krks_profile hcore --cell diamond --basis gth-dzvp \
    --nk 3,3,3 --mesh 41,41,41 --reps 2 --arm "$arm"
done

# C6  formatting — ONLY files you touched (R9).
#     Compare against the committed version to separate your hunks from
#     pre-existing ones; the tree predates the installed rustfmt.
rustfmt --edition 2024 --check <file> | grep -c "^Diff in"
git show HEAD:<file> > /tmp/orig.rs && rustfmt --edition 2024 --check /tmp/orig.rs | grep -c "^Diff in"
# The two counts must be EQUAL. If yours is higher, fix only your own hunks by hand.
```

### 0.5 STOP CONDITIONS — halt and report, do not work around

1. C1 reports **any** failure, or fewer than 450 passed.
2. C2 reports any warning.
3. C3 fails.
4. A cubecl build error occurs and R3's guide does not cover it.
5. An instruction below names a file, symbol or line that does not exist.
6. A bitwise test in this plan cannot be made to pass. **Do not relax it to a
   tolerance.** See §2.4 for why bitwise is achievable here and where it is not.
7. You are about to block a reduction that uses `oracle_sum`. See §2.4. That case is
   **proven unachievable** and is out of scope.

---

## 1. STATE CHECK — run this first, every time

**DO.** Run each command. Use the table to pick the next item.

```bash
grep -c 'fn accumulator_only'                     crates/pyscf-pbc-df/tests/hcore_peak_rss.rs   # B-00
grep -c 'let zeros = vec!\[0.0f64'                crates/pyscf-kernels/src/pbc/eval_ao_k.rs     # B-01
grep -c 'AO_IMAGE_BATCH_BUDGET_BYTES / block_bytes' crates/pyscf-pbc-gto/src/eval_gto.rs        # B-02
grep -c 'PYSCF_PBC_AO_GRID_BLOCK'                 crates/pyscf-pbc-gto/src/eval_gto.rs          # B-03
```

| command | prints | meaning | next item |
|---|---|---|---|
| B-00 | `0` | not landed | do **B-00** |
| B-00 | `1` or more | landed | continue |
| B-01 | `1` | not landed | do **B-01** |
| B-01 | `0` | landed | continue |
| B-02 | `1` | not landed | do **B-02** |
| B-02 | `0` | landed | continue |
| B-03 | `0` | not landed | do **B-03** |
| B-03 | `1` or more | landed | do **B-04** |

Values as of 2026-09-21 (nothing landed): `0`, `1`, `1`, `0`.

---

## 2. BACKGROUND — WHY. Not instructions. Skip if executing.

### 2.1 The measurement that produced this plan

K-14f fused `get_hcore`'s local contraction onto the device, removing the AO table's
read-back and per-k host planes. It did **not** remove the peak. Measured 2026-09-21,
diamond `gth-szv` (nao 8), 16 cores, CubeCL CPU runtime, peak RSS above baseline, one
call per process:

| nk, mesh | table | `ao_table_only` | host `get_pp` | fused `get_pp` |
|---|---|---|---|---|
| 2³, 31³ | 29.1 MiB | +183.0 MiB | +199.6 MiB | +169.0 MiB |
| 2³, 41³ | 67.3 MiB | +261.4 MiB | +272.5 MiB | +221.8 MiB |
| 3³, 41³ | 227.2 MiB | +790.5 MiB | +800.7 MiB | +588.3 MiB |

`ao_table_only` is the AO evaluation and read-back with **no contraction at all**, and
it already costs 3.48× the table. The fused route took `get_pp` from 3.52× to 2.59×.
**2.59× remains, all of it inside the evaluation.**

### 2.2 Where the 2.59× sits — ARITHMETIC FROM SOURCE, NOT MEASURED

At 3³/41³ (`nkpts` 27, `nao` 8, `ngrids` 68 921, `comp` 1, table 227.2 MiB):

| term | formula | value | verdict |
|---|---|---|---|
| accumulator planes | `2 · nkpts · n · 8` | 227.2 MiB | inherent — B-03 targets it |
| the zeros `Vec` they are uploaded from | `nkpts · n · 8` | 113.6 MiB | pure waste — B-01 removes it |
| K-09 image batch | `cap · n · 8`, `cap = min(32, 256 MiB / n·8)` | 134.4 MiB | fixed budget — B-02 fixes it |

Sum 475.2 MiB of the measured 588.3 MiB. The rest is grid coordinates
(`3·ngrids·8` = 1.6 MiB), the shift workspace, `pp_local_potential_r`'s `coulG`/`SI`/
`vlocG`, and allocator slack.

**B-00 exists because this table is arithmetic, not measurement.** Every item is
justified by one of its rows, so it is measured before it is acted on.

### 2.3 Modelled effect of B-03

`BLK = 8192` at 3³/41³: accumulator 227.2 MiB → 27.0 MiB. **Modelled, not measured.**

### 2.4 Why bitwise is achievable here — and exactly where it is not

The K-14f contraction accumulates `sr += term(g)` with a **serial** loop over `g` in
increasing order, one lane per `(k, p, q)`. A running serial sum over
`[0,b₁) ∪ [b₁,b₂)` using **one carried accumulator** is the identical sequence of IEEE
additions as a running serial sum over `[0,b₂)`. So blocking is bit-exact **provided**:

* blocks are visited in increasing `g` order,
* serially — the block loop is **not** parallelised over the output accumulator,
* the output accumulator is **carried**, never per-block-then-merged.

Summing each block independently and adding the partials afterwards changes the
association and **will** move the last bits.

**WHERE THIS DOES NOT HOLD.** `oracle_sum` is a pairwise tree whose shape is a function
of input **length**, so `oracle_sum([oracle_sum(b₀), …])` is a different tree from
`oracle_sum(b₀ ++ b₁ ++ …)` for any partition with more than one block. KRKS-plan
erratum E-5 proved a bit-identity contract across block sizes **unachievable** for
`KNumInt::nr_rks` for exactly this reason, and settled for 1e-13 relative there.

**This plan's bitwise contract covers ONLY the `local_vmat` path, which uses a serial
accumulator.** If you find yourself blocking an `oracle_sum` consumer, that is STOP
condition §0.5.7.

---

## 3. ITEMS

---

## B-00 — Measure the three terms of §2.2

**GOAL.** Turn §2.2's arithmetic into measurement. One number per term.

**PRECONDITION.** `grep -c 'fn accumulator_only' crates/pyscf-pbc-df/tests/hcore_peak_rss.rs` prints `0`.

**FILE.** `crates/pyscf-pbc-df/tests/hcore_peak_rss.rs`

Existing structure you will extend (line numbers as of 2026-09-21):

| line | item |
|---|---|
| 26 | `const MESH: [usize; 3]` |
| 29 | `fn triple(var, default) -> [usize; 3]` |
| 60 | `fn reset_peak() -> bool` |
| 64 | `struct Probe` |
| 71 | `Probe::start(label, nkpts, nao, ngrids)` |
| 80 | `Probe::finish(self)` |
| 97 | `fn setup() -> (Cell, Vec<[f64;3]>, [usize;3])` |
| 107 | `fn ao_table_only()` — copy this one |
| 119 | `fn host_route()` |
| 133 | `fn fused_route()` |

**DO.**

1. Add a fourth probe, modelled exactly on `ao_table_only` (line 107):

```rust
/// The accumulator alone — its two device planes AND the host zeros `Vec` they
/// are uploaded from. B-01 targets the second of those.
#[test]
#[ignore = "instrument, not a gate"]
fn accumulator_only() {
    let (cell, kpts, mesh) = setup();
    let df = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("FFTDF");
    let (nkpts, nao, ngrids) = (kpts.len(), cell.mol.nao_nr, df.ngrids());
    let p = Probe::start("accumulator_only", nkpts, nao, ngrids);
    let acc = pyscf_kernels::pbc::AoKAccumulator::zeros(
        &pyscf_algebra::select_backend().expect("backend").client,
        nkpts,
        nao * ngrids,
    );
    std::hint::black_box(&acc);
    p.finish();
}
```

2. Add a fifth probe `image_batch_only`, same shape, calling
   `pyscf_kernels::pbc::AoImageBatch::new(&client, capacity, n, ngrids, nkpts)` with
   `capacity = 32` (the value `image_batch_capacity` yields at this shape) and
   `n = nao * ngrids`.
3. **No `Cargo.toml` change is needed.** `pyscf-kernels` and `pyscf-algebra` are
   already regular `[dependencies]` of `pyscf-pbc-df`, and Cargo makes a package's
   regular dependencies available to its integration tests. `tests/hcore_fused.rs`
   already does `use pyscf_algebra::CTensor;` on that basis. Do not add dev-deps.
4. Run **C4**, extended to the five probe names.
5. Append the five numbers to
   `.planning/phases/11-fft-fftdf-periodic-hf/measurements/k14f-hcore-fused-contraction.md`
   as a new `## B-00 — measured term split` section.
6. In that section write one literal sentence: `§2.2's arithmetic HELD.` or
   `§2.2's arithmetic DID NOT HOLD: <term> measured <x> MiB against <y> MiB predicted.`

**DONE WHEN.**
- [ ] `grep -c 'fn accumulator_only'` prints `1`.
- [ ] `grep -c 'fn image_batch_only'` prints `1`.
- [ ] C4 prints five probe results.
- [ ] The measurements file contains the sentence from step 6.
- [ ] C1 still prints `450 passed; 0 failed` — the new probes are `#[ignore]`d and must
      **not** raise the passed count.

**ABORT IF.** The new probes are not `#[ignore]`d, or C1's passed count changes.

**WHY.** §2.2.

---

## B-01 — Delete the host zeros `Vec` in `AoKAccumulator::zeros`

**GOAL.** Remove `nkpts · n · 8` bytes (113.6 MiB at 3³/41³) of host allocation whose
only purpose is to be zero.

**PRECONDITION.** `grep -c 'let zeros = vec!\[0.0f64' crates/pyscf-kernels/src/pbc/eval_ao_k.rs` prints `1`.

**FILES.**
- `crates/pyscf-kernels/src/pbc/eval_ao_k.rs` — `AoKAccumulator::zeros`, line 588.
- `crates/pyscf-kernels/src/pbc/local_vmat.rs` — `zero_imag_kernel`, line 172.

**DO.**

1. Read `crates/pyscf-kernels/src/pbc/local_vmat.rs` lines 166–177. That is the fill
   kernel you will reuse. It is currently:

```rust
#[cube(launch_unchecked)]
fn zero_imag_kernel<F: Float>(im: &mut Array<F>, base: usize, stride: usize, n: usize) {
    let i = ABSOLUTE_POS;
    if i < n {
        im[base + i * stride] = F::from_int(0);
    }
}
```

2. Move it to a new file `crates/pyscf-kernels/src/pbc/fill.rs`, rename it
   `fill_zero_kernel`, make it `pub(crate)`, and add a `pub(crate)` host launcher
   `fill_zero<R: Runtime, F: DeviceScalar>(client, handle, len)` that calls it with
   `base = 0`, `stride = 1`, `n = len`. Use `launch_1d(client, len, 1)` for the
   geometry, exactly as `zero_gamma_plane` does today.
3. Add `pub mod fill;` to `crates/pyscf-kernels/src/pbc/mod.rs`, in alphabetical
   position (after `ewald`, before `ft_aopair`).
4. In `local_vmat.rs`, delete `zero_imag_kernel` and call the moved kernel instead.
   **Do not write a second fill kernel.**
5. In `eval_ao_k.rs`, replace `AoKAccumulator::zeros` lines 594–600 exactly:

```rust
// BEFORE
let zeros = vec![0.0f64; (nkpts * n).max(1)];
let (re, im) = dispatch_backend!(
    client,
    c,
    Rt,
    (upload(c, zeros.as_slice()), upload(c, zeros.as_slice()))
);

// AFTER
// `max(1)`: a zero-length allocation is not something every cubecl backend is
// obliged to handle. T4 — `empty` hands back a DIRTY recycled buffer, so the
// fill is mandatory, not an optimisation.
let len = (nkpts * n).max(1);
let bytes = len * core::mem::size_of::<f64>();
let (re, im) = dispatch_backend!(client, c, Rt, {
    let re = c.empty(bytes);
    let im = c.empty(bytes);
    crate::pbc::fill::fill_zero::<Rt, f64>(c, &re, len);
    crate::pbc::fill::fill_zero::<Rt, f64>(c, &im, len);
    (re, im)
});
```

6. Keep the `(nkpts * n).max(1)` guard. Do not remove it.
7. Add this test to `crates/pyscf-kernels/tests/pbc_eval_ao_k.rs`:

`pbc_eval_ao_k.rs` has NO `cpu_client()` helper — it builds a client with
`select_backend()` (already imported at line 14). Use that idiom exactly:

```rust
/// T4 — `client.empty` recycles dirty buffers, so a second accumulator built
/// after the first is dropped must still read back all-zero.
#[test]
fn a_recycled_accumulator_is_still_zero() {
    let client = select_backend().expect("backend must resolve").client;
    let (nkpts, n) = (4usize, 64usize * 1024);
    drop(pyscf_kernels::pbc::AoKAccumulator::zeros(&client, nkpts, n));
    let acc = pyscf_kernels::pbc::AoKAccumulator::zeros(&client, nkpts, n);
    let (re, im) = acc.into_planes(&client);
    assert!(re.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()), "re not zero");
    assert!(im.iter().all(|v| v.to_bits() == 0.0_f64.to_bits()), "im not zero");
}
```

8. Run C1, C2, C3, C6.
9. Run C4. Record the new `accumulator_only` number beside B-00's.

**DONE WHEN.**
- [ ] `grep -c 'let zeros = vec!\[0.0f64' crates/pyscf-kernels/src/pbc/eval_ao_k.rs` prints `0`.
- [ ] `grep -rc 'fn zero_imag_kernel' crates/pyscf-kernels/src/` prints `0`.
- [ ] `a_recycled_accumulator_is_still_zero` passes.
- [ ] C1 prints `451 passed; 0 failed` (450 + this one new test).
- [ ] C2, C3 clean. C6 hunk counts equal.
- [ ] `accumulator_only` dropped by approximately `nkpts · n · 8` bytes.

**ABORT IF.** Any element of a fresh accumulator reads non-zero. That means the fill
did not cover the buffer — **do not** paper over it by reverting to `upload`.

**BIT-PARITY.** Bitwise. A buffer of literal `0.0` is a buffer of literal `0.0`
however it was produced. `tests/hcore_fused.rs` proves it end to end.

**WHY.** `zeros` currently builds `vec![0.0f64; (nkpts * n).max(1)]` on the host and
uploads it **twice**. Its own comment argues the upload is "one transfer for the whole
loop either way" — true of the transfer, false of the allocation.

---

## B-02 — Make the image-batch budget relative, not a fixed 256 MiB

**GOAL.** Stop the K-09 batch from being sized by an absolute constant that is now up
to 1.2× the AO table, and express it per **block** so B-03 shrinks it too.

**PRECONDITION.** `grep -c 'AO_IMAGE_BATCH_BUDGET_BYTES / block_bytes' crates/pyscf-pbc-gto/src/eval_gto.rs` prints `1`.

**FILE.** `crates/pyscf-pbc-gto/src/eval_gto.rs` — `image_batch_capacity`, line 995;
`AO_IMAGE_BATCH_BUDGET_BYTES`, line 1156.

Current body (lines 995–1004):

```rust
fn image_batch_capacity(block_len: usize) -> usize {
    if let Some(v) = std::env::var("PYSCF_PBC_AO_IMAGE_BATCH")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        return v.clamp(1, pyscf_kernels::pbc::AO_IMAGE_BATCH_MAX);
    }
    let block_bytes = block_len.saturating_mul(core::mem::size_of::<f64>()).max(1);
    (AO_IMAGE_BATCH_BUDGET_BYTES / block_bytes).clamp(1, pyscf_kernels::pbc::AO_IMAGE_BATCH_MAX)
}
```

**DO.**

1. Change the signature to `fn image_batch_capacity(block_len: usize, accumulator_bytes: usize) -> usize`.
2. Keep the `PYSCF_PBC_AO_IMAGE_BATCH` early return **unchanged**. It stays the
   absolute override.
3. Replace the last line with a budget that is the smaller of the absolute cap and a
   fraction of the accumulator:

```rust
let block_bytes = block_len.saturating_mul(core::mem::size_of::<f64>()).max(1);
// The batch used to be capped only by an absolute 256 MiB, chosen when it was
// the only large buffer in the call. It is now up to 1.2x the AO table, and
// after B-03 it would be the LARGEST buffer in a blocked evaluation. Tie it to
// the accumulator so blocking shrinks both.
let budget = AO_IMAGE_BATCH_BUDGET_BYTES
    .min(accumulator_bytes.saturating_mul(AO_IMAGE_BATCH_ACC_NUM) / AO_IMAGE_BATCH_ACC_DEN);
(budget / block_bytes).clamp(1, pyscf_kernels::pbc::AO_IMAGE_BATCH_MAX)
```

4. Add next to `AO_IMAGE_BATCH_BUDGET_BYTES`:

```rust
/// The image batch's budget as a fraction of the accumulator: NUM/DEN.
/// Provisional at 1/2 — B-04 replaces it with a measured value. Do not treat
/// this as tuned.
const AO_IMAGE_BATCH_ACC_NUM: usize = 1;
const AO_IMAGE_BATCH_ACC_DEN: usize = 2;
```

5. Update the one call site. Find it with
   `grep -n 'image_batch_capacity(' crates/pyscf-pbc-gto/src/eval_gto.rs`. Pass
   `accumulator_bytes = 2 * nkpts * n_expected * 8`.
6. Run C1, C2, C3, C6.
7. Run C4. Record the new `image_batch_only` number.

**DONE WHEN.**
- [ ] `grep -c 'AO_IMAGE_BATCH_BUDGET_BYTES / block_bytes'` prints `0`.
- [ ] `grep -c 'AO_IMAGE_BATCH_ACC_NUM'` prints `2` (definition + use).
- [ ] C1 prints `451 passed; 0 failed`.
- [ ] `PYSCF_PBC_AO_IMAGE_BATCH=1` still forces capacity 1 — verify by running the
      existing batch-invariance test in `crates/pyscf-kernels/tests/pbc_eval_ao_k.rs`
      with that variable set.

**BIT-PARITY.** Bitwise at any capacity, and already gated. Each `(k, p)` accumulator
receives exactly one `pr[k] · ao[q]` addition per image, in `Ls` order, whatever the
batch size. `PYSCF_PBC_AO_IMAGE_BATCH=1` is the per-image reference arm.

**WHY.** §2.2 row 3.

---

## B-03 — Block the grid in `eval_ao_kpts_accumulate`

**GOAL.** Make the accumulator `16 · nkpts · nao · BLK` instead of
`16 · nkpts · nao · ngrids`.

**PRECONDITION.** B-00, B-01 and B-02 are all landed per §1.

**THE PRICE — READ BEFORE STARTING.** The lattice-image loop runs once **per block**.
At the measured shape (`nimgs` = 1331, `ngrids/BLK` = 9 blocks) that is **11 979 image
iterations instead of 1 331**. This item can be a net time **loss**. B-04 is the
decision gate. Do not assume it lands.

**FILES.**
- `crates/pyscf-pbc-gto/src/eval_gto.rs` — `eval_ao_kpts_accumulate` (line 367),
  `eval_ao_kpts_local_vmat` (line 914), `SCREEN_BLKSIZE` (line 1303, value `128`).
- `crates/pyscf-kernels/src/pbc/local_vmat.rs` — `local_vmat_kernel` (line 124),
  `run` (line 490).

Split into five sub-steps. **Complete and verify each before starting the next.**

### B-03a — Add the block parameter, default to one block

**DO.**
1. Change `eval_ao_kpts_accumulate`'s signature (line 367) by appending one parameter:

```rust
fn eval_ao_kpts_accumulate(
    cell: &Cell,
    eval_name: &str,
    coords: &[[f64; 3]],
    kpts: &[[f64; 3]],
    ls: &[[f64; 3]],
    /// Grid range this call covers: `coords[g0..g1]`. `0..coords.len()` is the
    /// whole grid and reproduces the pre-B-03 behaviour exactly.
    grid: std::ops::Range<usize>,
) -> Result<AoAccumulated, PyscfRsError> {
```

2. Inside, replace every use of `coords` with `&coords[grid.clone()]` and every use of
   `ngrids` with `grid.len()`.
3. Update both call sites to pass `0..coords.len()`. Find them with
   `grep -n 'eval_ao_kpts_accumulate(' crates/pyscf-pbc-gto/src/eval_gto.rs`.
4. Run C1.

**DONE WHEN.** C1 prints `451 passed; 0 failed`. **No behaviour has changed yet** —
this sub-step is a pure refactor and the suite must be untouched.

### B-03b — Add the accumulate mode to the contraction kernel

**DO.**
1. In `local_vmat.rs`, append one parameter to `local_vmat_kernel` (line 124):
   `accumulate: u32`.
2. Replace the two writes at the end of the kernel body:

```rust
// BEFORE
out_re[i] = sr;
out_im[i] = si;

// AFTER
// `accumulate == 1` carries the sum across grid blocks (B-03). §2.4: the
// accumulator must be CARRIED, never per-block-then-merged.
if accumulate == 1 {
    out_re[i] += sr;
    out_im[i] += si;
} else {
    out_re[i] = sr;
    out_im[i] = si;
}
```

3. Thread `accumulate` through `launch_range` and `launch_on_handles`. Pass `0` from
   every existing call site.
4. In `run` (line 490), when `accumulate == 1` the output buffers must start zeroed —
   call B-01's `fill_zero` on them instead of leaving `client.empty` (T4).
5. Run C1 and the K-14f bitwise gates specifically:
   `cargo test --release -p pyscf-kernels --test pbc_local_vmat -p pyscf-pbc-df --test hcore_fused`.

**DONE WHEN.** C1 unchanged at `451 passed`. All 12 K-14f tests pass. Behaviour is
still identical because every call site passes `accumulate = 0`.

### B-03c — Add the blocked driver

**DO.**
1. Add `pub fn eval_ao_kpts_local_vmat_blocked(cell, coords, kpts, vr, blk: usize)`
   beside `eval_ao_kpts_local_vmat` (line 914).
2. Hoist OUT of the block loop, computed once:
   - the image list `ls` and its norm sort,
   - the `bloch_phase` table,
   - the `EvalGtoDeviceContext`,
   - the screen's block boxes and per-shell radii.

   **Re-uploading any of these per block makes B-03 a guaranteed loss.** See
   `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/11_launch_overhead_and_transfers.md`
   §2.
3. The block loop, exactly this shape:

```rust
// §2.4: increasing g order, SERIAL, one carried output accumulator.
// Do NOT parallelise this loop. Do NOT sum blocks independently and merge.
let mut g0 = 0usize;
while g0 < ngrids {
    let g1 = (g0 + blk).min(ngrids);
    let acc = eval_ao_kpts_accumulate(cell, "GTOval_sph", coords, kpts, &ls, g0..g1)?;
    // contract coords[g0..g1] into the CARRIED output with accumulate = 1
    g0 = g1;
}
```

4. `blk` must be a multiple of `SCREEN_BLKSIZE` (= 128). Assert it:
   `assert!(blk % SCREEN_BLKSIZE == 0)`. The W-09 screen already partitions the grid
   into 128-point chunks via `block_boxes`; reuse that partition, do **not** introduce
   a second one.
5. Run C1.

**DONE WHEN.** C1 unchanged. The new function exists but nothing calls it yet.

### B-03d — The bitwise gate

**DO.** Create `crates/pyscf-pbc-df/tests/hcore_block.rs`.

1. Copy `static FUSE_SWITCH` (`crates/pyscf-pbc-df/tests/hcore_fused.rs` line 44)
   and `fn with_fuse` (line 50), up to but excluding `const HOST` (line 70). T5. Do
   not write a new pattern.
2. Assert `get_pp` is **bit-identical** across `PYSCF_PBC_AO_GRID_BLOCK` ∈
   `{whole-grid, 8192, 1024, 128}`, comparing with `to_bits()`, for each of:
   - a gamma-only k-list (`[[0.0; 3]]`),
   - a 2×2×2 k-mesh,
   - a pseudopotential cell (`common::diamond()`),
   - an all-electron cell (`common::he_all_electron()`).
3. Run the new test.

**DONE WHEN.** Every combination is bit-identical.

**ABORT IF.** Any combination differs. Check §2.4's three conditions in order: blocks
in increasing `g` order; block loop serial; accumulator carried. **Do not relax the
test to a tolerance** — that is STOP condition §0.5.6.

### B-03e — Wire the switch

**DO.**
1. Read `PYSCF_PBC_AO_GRID_BLOCK` in `eval_gto.rs`. Rules, exactly:

| value | behaviour |
|---|---|
| unset | one block — today's behaviour |
| `0` | one block |
| a positive multiple of 128 | that `BLK` |
| anything else | one block, and emit `tracing::warn!` naming the bad value |

2. Run C1, C2, C3, C6, C4, C5.

**DONE WHEN.**
- [ ] `grep -c 'PYSCF_PBC_AO_GRID_BLOCK' crates/pyscf-pbc-gto/src/eval_gto.rs` ≥ 1.
- [ ] C1 passes. C2, C3 clean. C6 hunk counts equal.
- [ ] C4 and C5 recorded for `BLK` ∈ {whole-grid, 8192, 1024}.

**RISK — the point-major layout.** The K-10v fused AO path accumulates point-major
(`plane[e·nkpts + k]`), which strides the `g` walk by `nkpts`. K-14f measured that at
1.8× slower and works around it with a per-k gather. Blocking makes the gather scratch
`nao · BLK` (smaller) but it still runs `nkpts` times **per block**. Measure this arm
specifically with `PYSCF_PBC_AO_FUSE=1` and `=0`; it may be what sets `BLK`.

---

## B-04 — Decide, or refute in writing

**GOAL.** Convert B-03's measurements into a default, an opt-in, or a written refusal.

**PRECONDITION.** B-03e is landed.

**DO.**

1. Run C4 and C5 at all three shapes of §2.1, for `BLK` ∈ {whole-grid, 8192, 1024}.
2. Compute, against the whole-grid arm: `mem_saving = 1 − peak(BLK)/peak(whole)` and
   `time_cost = time(BLK)/time(whole) − 1`.
3. Apply this table. It is exhaustive — do not add a fourth case.

| condition | action |
|---|---|
| `mem_saving > 0.10` **and** `time_cost ≤ 0.10` | Default ON. Derive `BLK` from device cache properties. |
| `mem_saving > 0.10` **and** `time_cost > 0.10` | Keep OFF by default, opt-in via `PYSCF_PBC_AO_GRID_BLOCK`. Record the exchange rate. |
| `mem_saving ≤ 0.10` | **REFUTE.** Write the numbers into the measurements file under `## B-03 REFUTED`. Stop. |

4. Whichever row applies, append the numbers and the verdict to
   `.planning/phases/11-fft-fftdf-periodic-hf/measurements/k14f-hcore-fused-contraction.md`.
5. Replace B-02's provisional `AO_IMAGE_BATCH_ACC_NUM/DEN` with the measured value.

**DONE WHEN.** The measurements file contains one of the three verdicts, with numbers.

**WHY A REFUTATION MUST BE WRITTEN DOWN.** An undocumented refuted lever gets
re-proposed. This tree has that failure mode on record: the multigrid host contraction
was refuted at ≤1.08× and needed a written entry to stay refuted.

---

## 4. ORDER

```
B-00                    <- gates everything; §2.2 is arithmetic until it runs
 ├─ B-01                <- independent, pure win
 └─ B-02                <- independent, pure win
      └─ B-03a → B-03b → B-03c → B-03d → B-03e   <- strictly sequential
           └─ B-04
```

B-01 and B-02 may land in either order and are safe to ship alone. **B-03 must not
start before B-00 reports**, because §2.2 is arithmetic and B-03 is only worth its
price if the accumulator really is the dominant remaining term.

---

## 5. DEFINITION OF DONE — the whole plan

- [ ] §1 state check prints `1`, `0`, `0`, `1`.
- [ ] C1 passes with the expected count.
- [ ] C2 and C3 clean.
- [ ] `crates/pyscf-pbc-df/tests/hcore_block.rs` exists and every case is bitwise.
- [ ] The measurements file contains B-00's term split and B-04's verdict.
- [ ] No file outside these was modified:
      `crates/pyscf-kernels/src/pbc/{eval_ao_k,local_vmat,fill,mod}.rs`,
      `crates/pyscf-kernels/tests/pbc_eval_ao_k.rs`,
      `crates/pyscf-pbc-gto/src/eval_gto.rs`,
      `crates/pyscf-pbc-df/tests/{hcore_peak_rss,hcore_block}.rs`,
      `.planning/phases/11-fft-fftdf-periodic-hf/measurements/k14f-hcore-fused-contraction.md`.

---

## 6. REFERENCE — cubecl manual sections

| section | needed for |
|---|---|
| [`11_launch_overhead_and_transfers.md`](../../../cubecl_manual/manual/Cubecl/11_launch_overhead_and_transfers.md) §2 | B-03c step 2 — hoisting invariant uploads |
| [`11_launch_overhead_and_transfers.md`](../../../cubecl_manual/manual/Cubecl/11_launch_overhead_and_transfers.md) §6 | R6 — re-measure after every change |
| [`13_memory_preallocation.md`](../../../cubecl_manual/manual/Cubecl/13_memory_preallocation.md) | B-01 — `empty` vs `create` |
| [`03_kernel_fusion.md`](../../../cubecl_manual/manual/Cubecl/03_kernel_fusion.md) | B-03 — the argument K-14f applied |
| [`07_memory_coalescing.md`](../../../cubecl_manual/manual/Cubecl/07_memory_coalescing.md) | B-03e — the point-major stride risk |

---

## 7. OPEN QUESTIONS — answer before the step that depends on each

| # | question | blocks | what is already known |
|---|---|---|---|
| Q1 | Should `KNumInt` share `BLK`? | B-04 only | **Do not assume they can share a constant.** E-5 (§2.4) proved `KNumInt::nr_rks` cannot hold a bit-identity contract across block sizes, because it reduces with `oracle_sum`. It settled for 1e-13 relative. This plan's contract does not transfer. |
| Q2 | Can a block's AO values be reused across `get_hcore`/`get_j`/`get_k`? | B-04 only | Blocking sharpens the cache-vs-recompute trade. `Fftdf::ao_table_fits_cache` (K-14f's routing predicate) would need rewriting per block. Out of scope here; raise it if B-03 defaults ON. |
| Q3 | Is `comp = 4` (deriv-1, the band path) in scope? | **B-03c** | `eval_ao_kpts_local_vmat` refuses `comp != 1` today. A deriv-1 table is 4× larger, so blocking would help it most. **Answer this before B-03c fixes an interface.** |

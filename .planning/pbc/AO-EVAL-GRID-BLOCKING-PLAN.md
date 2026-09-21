# AO-evaluation grid blocking — cutting the last copy of the AO table

**Created:** 2026-09-21
**Target:** `pyscf_pbc_gto::eval_gto::eval_ao_kpts_accumulate` and everything that
consumes its `AoKAccumulator` — `FFTDF::get_pp` / `get_nuc` / `get_hcore`,
`fft_jk::get_j_kpts` / `get_k_kpts`, `KNumInt`.
**Status:** draft — **no code written**. §2 is MEASURED (see §2.0 for the harness);
§3 is modelled and every number in it is labelled as such.
**Audience:** an execution agent that follows instructions literally and does NOT infer.

This plan is the follow-on flagged by K-14f
([`../phases/11-fft-fftdf-periodic-hf/measurements/k14f-hcore-fused-contraction.md`](../phases/11-fft-fftdf-periodic-hf/measurements/k14f-hcore-fused-contraction.md)).
K-14f removed the AO table's **read-back and per-k host planes** from `get_hcore`'s
local half. It did not remove the table itself, and the measurement that closed it
showed the table is not where the peak was: **the AO EVALUATION stage is**. This plan
is about that stage.

---

## 0. HOW TO EXECUTE THIS PLAN

Inherits every standing rule of [`PBC-MASTER-PLAN.md`](./PBC-MASTER-PLAN.md) §0 and of
`AGENTS.md`. The ones that bind hardest here:

* **RULE 4 — tests live in separate files.** No `mod tests` at the bottom of a
  production source file. Integration tests go in `crates/<crate>/tests/<name>.rs`.
* **RULE 5 — cubecl.** Read
  `/home/user/Documents/workspace/cubecl_manual/manual/Cubecl/INDEX.md` before touching
  any kernel. Kernels stay generic over the device float. On any cubecl build error,
  STOP and read the error guide directory
  (`manual/Cubecl/cubecl_error_solution_guide/`) — blind fixes are a protocol
  violation.
* **RULE 6 — the algebra wall (ALG-06).** Only `pyscf-algebra`, `pyscf-runtime` and
  `pyscf-kernels` may name a `cubecl-*` type. `AoKAccumulator`'s handles stay private;
  `pyscf-pbc-gto` drives the loop without naming a cubecl type today and must still do
  so afterwards. `cargo run -p xtask --bin check-dependency-wall` enforces it.
* **RULE O — measure, change ONE thing, re-measure.** These levers *move* the
  bottleneck. Every item below ends with a re-run of §2.0's harness.

Plus two rules this plan earns for itself, both from K-14f's measurement traps:

* **RULE M1 — a peak-RSS A/B is ONE CALL PER PROCESS.** CubeCL pools device buffers
  per process-global client, so a second arm (or a second rep) in the same process
  inherits the first one's pool and reports the pool's high-water mark. Two arms that
  genuinely differed by ~260 MiB both reported 2680 MiB under `--reps 2`. Reset the
  watermark with `std::fs::write("/proc/self/clear_refs", "5\n")`, never with
  `ulimit -v` (a virtual-size cap makes every CubeCL launch panic).
* **RULE M2 — a fresh `Fftdf` per rep.** A shared one lets reps 2+ hit the AO cache,
  so the arm reports a cached reduction (31 ms) as though it were the whole call
  (2 078 ms).

---

## 1. Scope

### 1.1 In scope

The single allocation chain inside `eval_ao_kpts_accumulate`
(`crates/pyscf-pbc-gto/src/eval_gto.rs`) that scales with `ngrids`:

| allocation | size | where |
|---|---|---|
| `AoKAccumulator`'s two device planes | `2 · nkpts · comp · ngrids · nao · 8` B | `pyscf-kernels/src/pbc/eval_ao_k.rs`, `zeros` / `zeros_point_major` |
| the host `vec![0.0f64; nkpts · n]` those planes are uploaded FROM | `nkpts · comp · ngrids · nao · 8` B | same, `zeros` |
| the K-09 image batch | `min(32, 256 MiB / block_bytes) · comp · ngrids · nao · 8` B | `AoImageBatch::new` |

### 1.2 Out of scope (non-goals)

* Changing the **result**. Every item is bit-parity-preserving; §4 is the proof
  obligation, not a hope.
* The contraction itself. K-14f owns it and is done.
* `eval_ao_kpts_upstream` (`eval_gto_upstream.rs`) — the bit-exact oracle route. It
  has its own blocking and is not touched.
* The strain-tensor family (`eval_strain_ao_kpts`) — different image list, different
  kernel, already routed around `eval_ao_kpts_accumulate`.
* GDF / RSDF / AFTDF. FFTDF's uniform grid only.

### 1.3 GATE A — accuracy must not regress

`crates/pyscf-pbc-dft/tests/gate.rs` and `crates/pyscf-pbc-df/tests/fftdf.rs` keep
passing at their **current** tolerances, unchanged. Plus the K-14f gates, which are
stricter than tolerance-based and are the real contract here:

| gate | file | criterion |
|---|---|---|
| `get_pp` / `get_nuc` / `get_hcore`, fused vs host route | `crates/pyscf-pbc-df/tests/hcore_fused.rs` | **bitwise** |
| `local_vmat` vs the host contraction | `crates/pyscf-kernels/tests/pbc_local_vmat.rs` | **bitwise** |

---

## 2. The measurement this plan starts from

### 2.0 Harness

`crates/pyscf-pbc-df/tests/hcore_peak_rss.rs` — `#[ignore]`d probes, one `--exact`
invocation per process (RULE M1), `PROBE_NK` / `PROBE_MESH` to sweep size:

```bash
for t in ao_table_only host_route fused_route; do
  PROBE_NK=3 PROBE_MESH=41 cargo test -p pyscf-pbc-df --release \
    --test hcore_peak_rss -- --ignored --nocapture --exact $t
done
```

and `krks_profile hcore --arm host|fused|auto` for the time A/B.

### 2.1 MEASURED 2026-09-21 — the peak is upstream of the read-back

diamond `gth-szv` (nao 8), 16 cores, CubeCL CPU runtime, peak RSS above the
process baseline:

| nk, mesh | AO table | `ao_table_only` | host `get_pp` | fused `get_pp` |
|---|---|---|---|---|
| 2³, 31³ | 29.1 MiB | +183.0 MiB | +199.6 MiB | +169.0 MiB |
| 2³, 41³ | 67.3 MiB | +261.4 MiB | +272.5 MiB | +221.8 MiB |
| 3³, 41³ | 227.2 MiB | +790.5 MiB | +800.7 MiB | +588.3 MiB |

Read the third row: `ao_table_only` is the AO evaluation and read-back with **no
contraction at all**, and it already costs **3.48×** the table. K-14f's fused route
takes `get_pp` from 3.52× down to **2.59×** — it removed ~1× the table and
**2.59× remains, all of it inside the evaluation**.

### 2.2 Where the 2.59× sits

Read off §1.1 at the 3³/41³ row (`nkpts` 27, `nao` 8, `ngrids` 68 921, `comp` 1,
table = `16·nkpts·nao·ngrids` = 227.2 MiB):

| term | formula | value | note |
|---|---|---|---|
| accumulator planes | `2 · nkpts · n · 8` | 227.2 MiB | = 1× table (two f64 planes *are* the complex table) |
| the zeros `Vec` uploaded from | `nkpts · n · 8` | 113.6 MiB | 0.5× table, **transient, and pure waste** |
| K-09 image batch | `cap · n · 8` | up to 256 MiB | `cap = min(32, 256 MiB / block_bytes)`; `block_bytes = n·8` = 4.2 MiB here, so `cap = 32` and the batch is 134.4 MiB |

227.2 + 113.6 + 134.4 = 475.2 MiB of the measured 588.3 MiB. The remainder is the
grid coordinates (`3·ngrids·8` = 1.6 MiB), the shifted-coordinate workspace, the
`coulG`/`SI`/`vlocG` of `pp_local_potential_r`, and allocator slack.

**The accumulator is the only one of the three that is inherent.** The zeros `Vec` is
an implementation detail of `zeros`. The image batch is a tunable that is currently
sized against a fixed 256 MiB budget rather than against the grid block.

### 2.3 What K-14f already proves about blocking

K-14f's contraction is **bit-identical** to the host loop because one lane owns one
`(k, p, q)` and walks `g` in a serial `0..ngrids` loop. That is the same property
grid blocking needs (§4), and the same file already carries the machinery for the
awkward layout case (`gather_k_kernel`). Blocking does not have to re-derive either.

---

## 3. Work items

Numbered **B-xx** (blocking). B-01 and B-02 are independent of the blocking itself and
should land first — they are cheap, they are pure wins, and they shrink the number
that B-03 is measured against.

### B-00 — Extend the harness to attribute the three terms

**FILES** `crates/pyscf-pbc-df/tests/hcore_peak_rss.rs`

**WHY** §2.2's split is **arithmetic from source, not measurement**. Every item below
is justified by one of those three terms, so the split has to be measured before it is
optimised — otherwise B-01/B-02/B-03 are aimed at a model.

**STEPS**

1. Add an `accumulator_only` probe: build the client, call
   `AoKAccumulator::zeros(&client, nkpts, n)`, `black_box` it, report. That isolates
   the planes **and** the zeros `Vec` in one number.
2. Add an `image_batch_only` probe over `AoImageBatch::new` at the same shape.
3. Report each probe as a multiple of the table, exactly as `Probe::finish` does now.
4. Record the results in
   `.planning/phases/11-fft-fftdf-periodic-hf/measurements/` as a new section, and
   state whether §2.2's arithmetic held.

**BIT-PARITY** N/A — instrument only, `#[ignore]`d.

**TEST** The probes are the test. They must not join the default `cargo test` run.

---

### B-01 — Stop materialising `nkpts · n` host zeros to allocate the planes

**FILES** `crates/pyscf-kernels/src/pbc/eval_ao_k.rs`

**WHY** `AoKAccumulator::zeros` builds `vec![0.0f64; (nkpts * n).max(1)]` on the HOST
and uploads it **twice**. That is 0.5× the table of host allocation (113.6 MiB at the
3³/41³ shape) whose only purpose is to be zero, plus two copies of it crossing to the
device. Its own doc comment argues the upload is "one transfer for the whole loop
either way" — true of the *transfer*, false of the *allocation*.

**STEPS**

1. Replace the two `upload(c, zeros.as_slice())` calls with `c.empty(bytes)` plus a
   fill kernel — a one-line `#[cube(launch_unchecked)]` that writes `F::from_int(0)`,
   which `local_vmat.rs::zero_imag_kernel` already is. Lift that kernel into a shared
   location rather than writing a second one.
2. Drop the host `Vec` entirely.
3. Keep the `(nkpts * n).max(1)` degenerate-shape guard; `empty(0)` is not something
   every backend is obliged to handle.

**BIT-PARITY** A buffer of literal `0.0` is a buffer of literal `0.0` however it was
produced. **Bitwise**, and `tests/hcore_fused.rs` proves it end to end.

**RISK** `client.empty` returns whatever the pool last held. Every element MUST be
written by the fill before any accumulate reads it. Do not skip the fill on the
assumption that a fresh allocation is zeroed — it is not, and a recycled dirty buffer
is a known trap in this tree.

**TEST** `crates/pyscf-kernels/tests/pbc_eval_ao_k.rs` — add a case asserting a
freshly built accumulator reads back all-zero at a shape large enough to force a pool
recycle (build one, drop it, build a second, assert).

---

### B-02 — Size the K-09 image batch against the grid block, not a fixed 256 MiB

**FILES** `crates/pyscf-pbc-gto/src/eval_gto.rs` (`image_batch_capacity`,
`AO_IMAGE_BATCH_BUDGET_BYTES`)

**WHY** `AO_IMAGE_BATCH_BUDGET_BYTES = 256 * 1024 * 1024` is an absolute cap chosen
when the batch was the only large buffer in the call. It is now up to 1.2× the AO
table on its own, and after B-03 it would be the **largest** buffer in a blocked
evaluation — the blocking would shrink the accumulator and leave the batch untouched.

**STEPS**

1. Make the budget relative: `min(AO_IMAGE_BATCH_BUDGET_BYTES, f · accumulator_bytes)`
   with `f` tunable and defaulted from a measurement, not a guess.
2. Keep `PYSCF_PBC_AO_IMAGE_BATCH` as the existing absolute override.
3. After B-03, the budget must be expressed per **block**, so write it in terms of the
   block length from the start.

**BIT-PARITY** K-09 batching is already documented as bit-identical at any capacity —
each `(k, p)` accumulator receives exactly one `pr[k] · ao[q]` addition per image, in
`Ls` order, whatever the batch size. `PYSCF_PBC_AO_IMAGE_BATCH=1` is the per-image
reference arm. **Bitwise**, and it is already gated.

**TEST** `crates/pyscf-kernels/tests/pbc_eval_ao_k.rs` already covers batch-size
invariance; extend its capacity list to include the new derived value.

---

### B-03 — Block the grid in `eval_ao_kpts_accumulate` — **THE ITEM**

**FILES** `crates/pyscf-pbc-gto/src/eval_gto.rs`,
`crates/pyscf-kernels/src/pbc/local_vmat.rs`

**WHY** The accumulator is `16 · nkpts · nao · ngrids` bytes and is the floor on every
consumer of the AO table. Blocking the grid into chunks of `BLK` points makes it
`16 · nkpts · nao · BLK`. At the 3³/41³ shape with `BLK = 8192` that is 227.2 MiB →
27.0 MiB (**modelled**, not measured).

**THE PRICE, STATED UP FRONT** the lattice-image loop runs once **per block**. The
per-image work that does not scale with the block — the Bloch phase lookup, the
launch, the per-image screen decision — is paid `nblocks` times instead of once. With
`nimgs = 1331` (measured at 3³/41³) and `ngrids/BLK = 9` blocks, that is 11 979 image
iterations instead of 1 331. **This item can easily be a net time LOSS.** It is worth
doing only if §2's harness says the memory matters more, and B-04 is the escape hatch.

**STEPS**

1. Give `eval_ao_kpts_accumulate` a block range parameter (`g0..g1`) and make it
   operate on `coords[g0..g1]`. The W-09 screen already partitions the grid into
   `SCREEN_BLKSIZE = 128` chunks (`block_boxes`), so **make `BLK` a multiple of
   `SCREEN_BLKSIZE`** and reuse that partition rather than introducing a second one.
2. Hoist everything image-invariant OUT of the block loop: the image list and its norm
   sort, the `bloch_phase` table, the `EvalGtoDeviceContext`, the screen's block boxes
   and per-shell radii. Only the accumulator and the per-block coordinate slice are
   per-block. Re-uploading any of these per block would make B-03 a guaranteed loss —
   see the cubecl manual's
   [`11_launch_overhead_and_transfers.md`](../../../cubecl_manual/manual/Cubecl/11_launch_overhead_and_transfers.md)
   §2.
3. Add a blocked consumer entry point beside `eval_ao_kpts_local_vmat`: it owns the
   `nkpts · nao²` output accumulator, loops blocks, and for each block runs the image
   loop then contracts that block's `g` range into the **carried** output.
4. `local_vmat_kernel` needs an `accumulate` mode (`out[i] += sr` instead of
   `out[i] = sr`) for step 3, and its output buffer must then start zeroed — use B-01's
   fill kernel, not `client.empty`.
5. Default `BLK` from the device's cache properties, overridable by
   `PYSCF_PBC_AO_GRID_BLOCK`. `0` or unset-and-unprofitable means "one block" — i.e.
   exactly today's behaviour, which is the reference arm.

**BIT-PARITY** **Bitwise, and this is the constraint that shapes the whole item.**

The contraction is `sr += term(g)` accumulated over `g` in increasing order. Splitting
`0..ngrids` into blocks preserves that sum **exactly** iff the accumulator is
*carried* across blocks and the blocks are visited in increasing `g` order — because a
running serial sum over `[0,b1) ∪ [b1,b2)` with one accumulator is the same sequence of
IEEE additions as a running serial sum over `[0,b2)`. It is **NOT** preserved if each
block is summed independently and the partials are added afterwards; that changes the
association and will move the last bits.

Therefore:

* blocks are visited in increasing `g` order, serially — **do not** parallelise the
  block loop over the output accumulator;
* the output accumulator is carried, never per-block-then-merged;
* the AO accumulator is per-block, which is fine: an AO value at grid point `g` is a
  sum over IMAGES, and the image order is unchanged within a block.

`crates/pyscf-pbc-df/tests/hcore_fused.rs` is the gate and it is already bitwise.

**RISK — the point-major layout.** The K-10v fused AO path accumulates point-major so
its k-loop vectorises, which strides the `g` walk by `nkpts`. K-14f measured that at
1.8× slower and works around it with a per-k gather. A block's accumulator is
`nkpts · nao · BLK`, so the gather scratch becomes `nao · BLK` — smaller, but the
gather still runs `nkpts` times **per block**. Measure this arm specifically; it may
be the thing that decides `BLK`.

**TEST** `crates/pyscf-pbc-df/tests/hcore_block.rs` (new) — `get_pp` must be
**bit-identical** across `PYSCF_PBC_AO_GRID_BLOCK ∈ {whole-grid, 8192, 1024, 128}`, at
a gamma-only k-list and at a 2×2×2 mesh, on both a pseudopotential and an
all-electron cell. Use the `with_fuse`-style process-wide lock from
`tests/hcore_fused.rs`: the switch is process-global and the harness runs test bodies
concurrently — this file first failed at `--test-threads=4` and passed at 2 for exactly
that reason.

---

### B-04 — The decision, and the escape hatch

**FILES** whichever of B-01..B-03 landed

**WHY** B-03 trades time for memory and the exchange rate is not known in advance.
RULE O says re-measure; this item says what to do with the answer.

**STEPS**

1. Re-run §2.0's harness after B-01, after B-02 and after B-03, one change at a time.
2. Record peak RSS **and** wall time for each, at all three shapes of §2.1.
3. Decide:
   * B-03 wins on memory and costs **≤ 10 %** time → default it on, with `BLK` from
     the device.
   * B-03 wins on memory and costs **more** → keep it **opt-in** behind
     `PYSCF_PBC_AO_GRID_BLOCK`, and document the exchange rate in the measurements
     note so the next reader does not re-litigate it.
   * B-03 does not win on memory → **REFUTE it in writing** in the measurements note,
     with the numbers, and stop. A refuted lever that stays undocumented gets
     re-proposed; this tree has that failure mode on record (the multigrid host
     contraction was refuted at ≤1.08× and needed a memory entry to stay refuted).

**BIT-PARITY** N/A — decision only.

**TEST** N/A.

---

## 4. Sequencing

```
B-00  (harness: attribute the three terms)      <- gates everything
  |
  +-- B-01  (drop the host zeros Vec)           independent, pure win
  +-- B-02  (relative image-batch budget)       independent, pure win
        |
        +-- B-03  (block the grid)              THE item; measured against B-01+B-02
              |
              +-- B-04  (decide / refute)
```

B-01 and B-02 may land in either order and are safe to ship on their own. B-03 must
not start before B-00 reports, because §2.2's split is arithmetic and B-03 is only
worth its price if the accumulator really is the dominant remaining term.

---

## 5. Verification protocol — run after EVERY item

1. `cargo test -p pyscf-kernels -p pyscf-pbc-gto -p pyscf-pbc-df --release`
   — the K-14f baseline is **450 passed / 0 failed / 63 ignored** (2026-09-21).
2. `cargo clippy -p pyscf-kernels -p pyscf-pbc-gto -p pyscf-pbc-df --tests` — CI runs
   it with `-D warnings`; `type_complexity` and `float_literal_f32_fallback` both bit
   K-14f.
3. `cargo run -p xtask --bin check-dependency-wall` — ALG-06.
4. §2.0's harness, one call per process (RULE M1), fresh `Fftdf` per rep (RULE M2).
5. `rustfmt --check` on the files you touched **only**. The tree predates the
   installed rustfmt, so a blanket `cargo fmt` reformats unrelated code; compare the
   hunk count against `git show HEAD:<file>` to tell your hunks from the pre-existing
   ones.

---

## 6. Risks

| risk | mitigation |
|---|---|
| B-03 is a net time loss | B-04 is the decision gate, and `PYSCF_PBC_AO_GRID_BLOCK` keeps whole-grid available. Measure before defaulting. |
| Block-independent accumulation silently changes the last bits | §B-03's BIT-PARITY: carry the output accumulator, blocks in increasing `g` order, serial. `tests/hcore_block.rs` is bitwise, not tolerance-based. |
| `client.empty` hands back a dirty recycled buffer | B-01's RISK note: fill before any read, and gate it with a build-drop-rebuild test. |
| Per-block re-upload of image-invariant data eats the win | §B-03 step 2 enumerates exactly what must be hoisted. |
| The point-major gather runs `nkpts` times per block | §B-03's RISK note; measure that arm specifically and let it set `BLK`. |
| A process-global env switch races across concurrent test bodies | Reuse the `FUSE_SWITCH` mutex pattern from `tests/hcore_fused.rs`. |

---

## 7. CubeCL manual sections this plan depends on

* [`11_launch_overhead_and_transfers.md`](../../../cubecl_manual/manual/Cubecl/11_launch_overhead_and_transfers.md)
  — §2 hoisting invariant uploads (B-03 step 2), §6 re-attributing after every change
  (RULE O).
* [`13_memory_preallocation.md`](../../../cubecl_manual/manual/Cubecl/13_memory_preallocation.md)
  — `empty` vs `create`, and why the hot loop should not reallocate (B-01, B-03).
* [`03_kernel_fusion.md`](../../../cubecl_manual/manual/Cubecl/03_kernel_fusion.md)
  — the fusion argument K-14f already applied; B-03 extends it across the block loop.
* [`07_memory_coalescing.md`](../../../cubecl_manual/manual/Cubecl/07_memory_coalescing.md)
  — the point-major stride risk in B-03.

---

## 8. Open questions

1. **Does `KNumInt` want the same blocking?** It has its own `block_ranges` and its own
   AO cache, and W-07 of the KRKS plan already touched them. If B-03 lands, the two
   blockings should agree on `BLK` rather than each picking its own — but W-07's
   erratum (E-5) recorded that its bit-identity criterion was unachievable there.
   Check E-5 before assuming the two can share a constant.
2. **Can a block's AO values be reused across consumers?** `get_hcore`, `get_j` and
   `get_k` each want the same table. Blocked evaluation makes the cache-vs-recompute
   trade sharper, and `Fftdf::ao_table_fits_cache` — which K-14f introduced as the
   routing predicate — would need rewriting per block. Not in scope here; flag it if
   B-03 defaults on.
3. **Is `comp = 4` (deriv-1) in scope?** The band path evaluates `GTOval_sph_deriv1`,
   whose table is 4× larger, so blocking would help it most. `eval_ao_kpts_local_vmat`
   refuses `comp != 1` today. Answer before B-03 step 3 fixes an interface.

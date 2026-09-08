# KRKS + k-symmetry + multigrid — session 5 execution record

**Plan:** [`KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md`](./KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md);
session 4's record: [`KUKS-KSYMM-MULTIGRID-SESSION-4-EXECUTION-SUMMARY.md`](./KUKS-KSYMM-MULTIGRID-SESSION-4-EXECUTION-SUMMARY.md),
whose §5 named the three levers this session takes in order.
**Date:** 2026-09-08 (same day, same box, same rules).
**Machine / load:** 16 cores, 30 GiB, CubeCL CPU runtime; the 1-minute load
at launch is inside every JSON and quoted beside every number (1.9-7.9 here,
the higher values being a series' own predecessors decaying).
**Manuals read before touching a kernel (RULE 5):** `11_launch_overhead_and_transfers.md`
§2/§5 (hoist uploads, collapse launches), `Cubecl_loop_control.md` (range
loops), `Backend-Agnostic_Buffer_Slicing…` (in-kernel offsets), `03_kernel_fusion.md`
(read for the lever this session does NOT take, §4). The error protocol's
guideline file (`manual/cubecl_error_guideline.md`) does not exist at the
path `AGENTS.md` names; the nearest documents are
`Cubecl/cubecl_error_solution_guide/*` (two items, neither covering §1.3's
error) and the loop-control manual, which is what §1.3 follows.

---

## 0. What this session did

| step | result |
|---|---|
| **A-05** (new): image-invariant AO operands (`env`, `bas`, `atm`, `ao_loc`, `rcut2`, nine angular tables) uploaded once per `eval_ao_kpts` (`EvalGtoDeviceContext`) | landed, bit-exact (same values); measured as the `EVAL_BATCH=1` arm, §1.2 |
| **A-06** (new): one AO evaluation launch per image batch, straight into the K-09 slots (`eval_gto_batch_into_image_batch`, three `*_batched` kernels) | landed, **bit-exact** (`eval_ao_image_batch` extended: eval batch 1 / 2 / 5 / 16 vs the per-image kernels), default on; **measured 1.01-1.05× on the cold pass** — the per-launch model was wrong, §1.2 |
| `AO_IMAGE_BATCH_MAX` 16 → 32 | landed; measured §1.4 |
| **M-14** (new): one output scratch per multigrid-v2 level instead of one per chunk (`PairOutScratch`, `PairSlotBatchDevice::new_shared`) | landed, bit-exact (`multigrid_batch`); RSS §2 |
| **K-10** (new, **D-PBC-32**, user-authorised over plan 10-04): the fused evaluate-and-accumulate AO kernel — no AO block ever written or read | landed, **bit-exact**, default on; **1.5-1.6× on the cold pass at 2×2×2, 1.32× on the KRKS SCF**, §1.5 |
| **`launch_1d_chunked`** (new, `pyscf-algebra`): chunked launches for kernels with per-lane local arrays | landed, bit-neutral; fixes a stack overflow that scales with lanes × locals on the CPU runtime, §1.5 |
| **K-10v** (new): the fused accumulate's k-loop as a `Vector<f64, N>` over a point-major accumulator, read back per k | landed, **bit-exact** (incl. ragged widths); **2.3× on the 4×4×4 cold pass, 1.73× on the 4×4×4 SCF**, §1.6 |
| three CubeCL CPU-runtime failures hit and resolved per `AGENTS.md` §4 | §1.3, §1.5 (×2) |

---

## 1. GATE AO

### 1.1 What session 4 left

After K-09 the cold pass at gth-szv deriv 1 was 2.36-2.49 s: ~0.9 s of AO
evaluation launches (the `SKIP_K08` arm) and the rest the batched
accumulate. Session 4 §1.2's floor arm (every lane skipping its arithmetic)
cost 0.4 / 1.0 s for 454 launches, which this session read as a
per-LAUNCH fixed cost — dispatch plus fourteen small uploads plus the output
allocation — and set out to remove with A-05 (hoist) and A-06 (collapse).

### 1.2 A-05 / A-06 measured — the per-launch model is refuted too

`krks_profile ao --cell si --mesh 31,31,31`, same binary, K-09 batch 16;
`EVAL_BATCH=0` = per-image kernels (session 4), `=1` = hoisted uploads with
one batched-kernel launch per image, default = one launch per 16 images
(`scratchpad/ab7`, load 3.0-7.9):

| row | `EVAL_BATCH=0` | `=1` (A-05 only) | default (A-05 + A-06) | ratio |
|---|---|---|---|---|
| gth-szv deriv 0, cold pass | 1 070 ms | 1 085 ms | 1 062 ms | 1.01 |
| gth-szv deriv 0, cold pass, `SKIP_K08` (the AO stage alone) | 586 ms | — | 533 ms | 1.10 |
| gth-szv deriv 1, cold pass | 2 493 ms | 2 497 ms | 2 487 ms | 1.00 |
| gth-szv deriv 1, `SKIP_K08` | 924 ms | — | 866 ms | 1.07 |
| gth-dzvp deriv 1, cold pass | 10 127 ms | — | 9 579 ms | 1.06 |
| gth-szv deriv 1, 4×4×4, cold pass | 10 521 ms | — | 10 078 ms | 1.04 |

Collapsing 454 launches into 29 and hoisting fourteen uploads per launch
moves the AO stage by 7-10 % and the cold pass by 0-6 %. So the 0.4-1.0 s
"floor" of session 4 is not launch overhead either: it is the lane body's
scaffolding on the CPU runtime — the `tid % ngrids` / `tid / ngrids` decode,
the `bas`/`atm`/`env` row loads, `r²`, the reach test and the output index
arithmetic — at ~150 ns per lane-iteration across 37.7 M lanes, with the
exponentials and the angular products adding only ~30 % on top (session 4
§1.2). A-06 stays on (bit-exact, ≥ 1.0 on every row, its staging is
`B · 3 · npts · 8` B), but it is recorded as **not a lever**; the RSS
column moved by +1 % (dzvp, 4×4×4) to +25 % (the 300 MiB deriv-0 process,
pool granularity).

### 1.3 The build-error protocol, applied

The first A-06 kernel located a lane's image with a `while` binary search
whose body was an `if/else`. Every batched arm aborted at run time inside the
CPU runtime's MLIR lowering:

```
error: operation with block successors must terminate its parent block
thread 'DSD-0-0' panicked at cubecl-cpu-0.10.0/src/compiler/module.rs:94:13:
failed to run pass
```

* **Root cause (VERIFIED by the fix, mechanism UNVERIFIED):** the
  `while { if … {} else {} }` shape; no other kernel in this crate has one
  (their `while`s are straight-line bodies).
* **Resolution:** a range `for` over the `nimg + 1` prefix table with a
  single `if` and no `else` (`eval_gto_image_of`), the shape
  `Cubecl_loop_control.md` recommends and the s-shell kernel already
  compiles. ≤ 32 compares per lane against hundreds of operations.
* **Verification:** the same binary ran every arm of §1.2 and the gate
  passed (§3).
* **Prevention:** recorded in the kernel's doc comment; `AGENTS.md` §4's
  guideline path is stale (see the header) and should be repointed.

### 1.4 `AO_IMAGE_BATCH_MAX = 32` — measured, kept

Same binary, `PYSCF_PBC_AO_IMAGE_BATCH=16` vs `=32`, A-06 on, load 3.1-3.5
(`scratchpad/s5`):

| row | `B = 16` | `B = 32` | ratio | peak RSS 16 → 32 |
|---|---|---|---|---|
| gth-szv deriv 1, 2×2×2, cold pass | 2 407 ms | **2 037 ms** | **1.18×** | 997 → 1 139 MiB |
| gth-szv deriv 0, 2×2×2 | 941 ms | 983 ms | 0.96× | 441 → 594 MiB |
| gth-szv deriv 1, **4×4×4** | 9 399 ms | **8 479 ms** | **1.11×** | 3 448 → 3 578 MiB |

Deriv 1 — the table every GGA SCF builds and the larger of the two — gains
11-18 %; deriv 0 is within noise. The cost is the staged blocks (`32 · 7.6`
MB) plus pool granularity, +130 to +150 MiB. The cap is 32 and the 256 MiB
budget stays the dial (`AO_IMAGE_BATCH_BUDGET_BYTES`; at gth-dzvp deriv 1 it
yields 10 images regardless). On a whole SCF (`krks_profile ksymm`, defaults,
load 3.4-3.9): KRKS si 2×2×2 PBE `kernel()` 4 534 → **4 261 ms** (full BZ),
3 863 → **3 620 ms** (ksymm), peak RSS 916 → 1 226 MiB; 4×4×4 15 027 →
**13 320 ms** / 13 329 → **12 284 ms**, RSS 3 654 → 4 104 MiB. Energies
identical to every printed digit (bit-exact by gate). The session-3 4×4×4
row was 45.8 / 43.1 s: **3.4× / 3.5×** over two sessions.

---

## 2. GATE MG — M-14, one output scratch per level

`crates/pyscf-kernels/src/multigrid_pair.rs` (`PairOutScratch`,
`PairSlotBatchDevice::new_shared`, `prefix`), `crates/pyscf-pbc-dft/src/multigrid/pair.rs`
(`PairLevelTable::out_scratch`, `resident_batch(lv, bl, client)`).

Each chunk owned its forward (`npoints · 8` B) and reverse (`nslots · 8` B)
output for the life of the SCF although chunks run sequentially and each
launch reads its output back before the next: at `25³` level 3, 13 chunks ×
48 MB of reverse output = 622 MB resident, live one chunk at a time. The
level now allocates its largest chunk's buffers once, every chunk writes and
reads its own prefix (`Handle::offset_end`), and — the session-3 pool lesson
— every chunk's geometry is allocated on the scratch's stream through
`StreamId::executes`, so one level lives on one stream whatever thread first
touched which chunk. Bit-exact: the same lanes write the same values to the
same logical positions.

`krks_profile multigrid --driver krks --numint v2 --mesh 25,25,25`, Si
gth-szv LDA (`scratchpad/s5/mg-v2-25-m14.json`, load 4.0):

| | session 3 (§5.2) | session 4, after M-13 | **session 5, after M-14** |
|---|---|---|---|
| peak RSS | 3 962 MiB | 3 070 MiB | **2 411 MiB** (−39 % over the two sessions) |
| `e_tot` | −7.160554062714283 | −7.160554062714283 | **−7.160554062714283** |
| warm `get_veff` | 9 842 ms | 10 035 ms | 10 254 ms (RULE M: speed untouched) |

The −659 MiB matches the model: level 3's 13 chunks × 48 MB + level 2's
4 × 50 MB of reverse output, minus the two shared maxima.

---

## 3. Gates re-run

No-LTO `target/gate` binaries, the same shape as sessions 3-4.

| gate | result |
|---|---|
| `pyscf-pbc-gto` `eval_ao_image_batch` (K-09 accumulate batch 1 / 3 / 16 × **A-06 eval batch 0 / 1 / 2 / 5 / 16** × **K-10/K-10v fused batch cap / 7 / 1**, `GTOval_sph` + `_deriv1`, gth-szv + gth-dzvp, screened in-process and unscreened in a child; plus the **ragged vector widths** `[1,2,3]` and `[1,1,3]`) | **3/3** — every real bit-identical to the per-image kernels, on the final (vectorised) build |
| `pyscf-pbc-gto` `eval_ao_point_screen` (A-04, off by default) | **2/2** — 2.440e-13 / 7.325e-13 against 1e-11, unchanged |
| `pyscf-pbc-gto` `eval_ao_stages` (GATE AO on the session-5 defaults: batch 32, A-06 on) | **2/2** |
| `pyscf-pbc-gto` `eval_ao_screen` (W-09) | **3/3** |
| `pyscf-pbc-dft` `multigrid_batch` (**M-14**: the batched arm now runs on the shared per-level scratch, vs streamed) | **4/4**, Si + diamond, 0e0 |
| `pyscf-pbc-dft` `multigrid_level_cache`, `multigrid_uks` | **1/1**, **5/5** |
| `pyscf-kernels`: `multigrid_pair` 8, `pbc_eval_ao_k` 7, `eval_gto_oracle` 3, `eval_gto_lge1` 4 | **22/22** |

The `pyscf-pbc-gto` and `pyscf-kernels` rows were re-run after K-10v (the
final build); the `pyscf-pbc-dft` rows date from the M-14 build (the only
later change under them is an ADDED function in `pyscf-algebra::launch`).

Not run: GATE A / GATE U (oracle venv), `multigrid2`, `multigrid_threads`,
`krks_ksymm`, `ksymm_*`, `numint_threads` — every item here is bit-exact by
its own gate above and the full-BZ oracle rows are unaffected by
construction; re-run them before the next baseline capture, as every session
of this plan has said of itself.

## 1.5 K-10 — the fused evaluate-and-accumulate kernel (**user-authorised, bit-exact**)

**Decision (2026-09-08, recorded here as D-PBC-32):** the user authorised
taking the fusion lever §4 below had set aside, i.e. lifting PBC-MASTER-PLAN
plan 10-04's "do not write a new AO evaluator" for the periodic AO table.
The per-image kernels (`eval_gto_sph_kernel*`), the K-08/K-09 accumulate and
the A-06 batched kernels all stay in the tree as the reference path
(`PYSCF_PBC_AO_FUSE=0`), and the fused kernel reuses their lane bodies
operand for operand — nothing about the AO arithmetic is new.

`crates/pyscf-kernels/src/eval_gto.rs` (`eval_ao_k_fused_kernel`,
`eval_gto_general_values` / `eval_gto_deriv1_values`, `AoGridDevice`,
`FusedImage`, `eval_ao_k_fused_batch`), `crates/pyscf-pbc-gto/src/eval_gto.rs`
(`FusedState`, `flush_fused`).

One lane per `(g, shell)` over the UNSHIFTED grid, uploaded once per call.
For each image `m` of the batch the lane forms `r_g − L_m` in-kernel (the
host's own subtraction, so the same bits), runs the per-image lane body into
a local value array, and then, per output `(c, ao)` of the shell and per
k-point, adds `pr[m,k]·v_m` into the resident planes IN IMAGE ORDER. A
point a screened image does not keep (`keep[m][block(g)]`, the W-09
decision) receives none from it. Each `(k, p)` therefore receives exactly
the additions the per-image path performed, in the same order — bit-exact,
asserted by `tests/eval_ao_image_batch.rs` (fused batch at the cap, at 7 and
at 1 image, both bases, both eval names, screened and unscreened) — and no AO
block is ever written or read. Per batch, what crosses to the device is
`3·B` lattice vectors, `B·nkpts` phases and `B·nblocks` keep flags; the K-09
slot buffer (up to 256 MiB) is not allocated at all on this path. The fused
batch is `FUSED_VALS_CAP / Q_max` images (32 at gth-szv, 12 at gth-dzvp
deriv 1); all-s bases at deriv 0 stay on K-09 (the s-kernel's arithmetic is
not the general kernel's).

**The build-error protocol, applied a second time.** The first fused build
aborted every run with

```
thread '<unknown>' has overflowed its stack
fatal runtime error: stack overflow, aborting
```

* **Diagnosis:** `CUBECL_CPU_STACK_MB=512` ran it, so the overflow is the
  CPU runtime's worker threads; `RUST_MIN_STACK` did nothing; the fused
  batch size did not matter; the threshold sat between 80 and 128 MB. The
  compiler puts every `LocalArray` alloca in the entry block
  (`cubecl-cpu/src/compiler/visitor/variables.rs:54-80`), so it is not a
  per-iteration alloca; the frame nevertheless scales with the local value
  array at ~50 KB per element (2 048 → >80 MB; 512 → <32 MB). Mechanism
  UNVERIFIED.
* **Resolution:** `FUSED_VALS_CAP` 2 048 → 512, which runs at the default
  64 MB and at 32 MB; the batch cap follows from it. Recorded in the
  constant's doc comment.
* **Verification / prevention:** the A/B below and the gate; no other kernel
  in the tree holds a local array above 32 elements, and any that will must
  probe `CUBECL_CPU_STACK_MB` first.

**The build-error protocol, applied a third time — and this one reaches
every kernel in the tree.** With the fused path in, the UNFUSED
`PYSCF_PBC_AO_FUSE=0` arm overflowed the worker stack on gth-dzvp deriv 1
and nowhere else. Bisected by switch: the per-image kernels
(`IMAGE_BATCH=1`) never fail; the K-09 accumulate kernel fails at gth-dzvp
whatever its batch (10 or 16) and had passed on the same fixture when its
local arrays were 16 wide; `CUBECL_CPU_STACK_MB=256` makes it run. The
numbers line up only one way: gth-dzvp deriv 1 has 3.1 M lanes, 194 k
iterations per unit on 16 units, × the 32-image arrays' 384 B = 75 MB > 64
MB; gth-szv's 0.95 M lanes give 23 MB; the fused kernel's 16 KB array over
7 400 iterations gives 122 MB (failed at 80, ran at 128 — §1.5's first
event, now explained). **CubeCL 0.10's CPU runtime consumes stack for a
kernel's `Array::new` locals per cube ITERATION**, and nothing releases it
until the launch ends; the lowering puts the alloca in the body's first
block, which sits inside the cube loop (`cubecl-cpu/src/compiler/visitor/variables.rs:54-80`).
Mechanism VERIFIED by the bisection, not by reading MLIR.

* **Resolution:** `pyscf_algebra::launch::launch_1d_chunked(client, lanes,
  work, local_bytes)` — on the CPU runtime the lane range is split into
  launches whose iterations per unit × `local_bytes` stay under
  `CPU_LOCAL_STACK_BUDGET` (16 MiB); a GPU gets one launch. The kernel adds
  the chunk's `lane0` to `ABSOLUTE_POS`. Every lane runs the same body on
  the same operands, so it is bit-neutral. Applied to the K-09 accumulate and
  the fused K-10 kernel; the multigrid v2 batched kernels (`Array::new(10)`,
  80-160 B) are unchunked and safe only while a chunk has under ~10 M
  instances — recorded, not changed (RULE O).
* **Verification:** the dzvp arms below and the gates (§3). Memory note
  `cubecl-cpu-local-arrays-cost-stack-per-iteration`.
* **Prevention:** the rule in that note — any kernel with `Array::new`
  locals launches through `launch_1d_chunked`.

**K-10 measured** — same binary, `PYSCF_PBC_AO_FUSE=0` (K-09 batch 32 +
A-06) vs fused, load 2.3-5.7 (`scratchpad/k10`, kept as
`baselines/2026-09-08-k10-*.json`; the machine rebooted once during this
block and the series was re-taken in full):

| row | unfused (K-09/A-06) | **fused K-10** | ratio | peak RSS unfused → fused |
|---|---|---|---|---|
| gth-szv deriv 0, 2×2×2, cold pass | 860 ms | **571 ms** | **1.51×** | 588 → 717 MiB |
| gth-szv deriv 1, 2×2×2 | 1 878 ms | **1 192 ms** | **1.58×** | 1 136 → 974 MiB |
| gth-dzvp deriv 1, 2×2×2 (chunked build) | 8 338 ms | **5 211 ms** | **1.60×** | 1 788 → 1 519 MiB |
| gth-szv deriv 1, 4×4×4 | 7 455 ms | 6 533 ms | 1.14× | 3 580 → 3 418 MiB |
| KRKS si 2×2×2 PBE `kernel()`, full BZ / ksymm | 3 833 / 3 337 ms | **2 898 / 2 313 ms** | **1.32× / 1.44×** | 1 222 → 1 092 MiB |
| KRKS si 4×4×4 PBE `kernel()`, full BZ / ksymm | 12 389 / 11 003 ms | 11 423 / 10 660 ms | 1.08× / 1.03× | 4 102 → 3 925 MiB |

Energies identical to every printed digit on every row (bit-exact by gate).
The 4×4×4 rows gain least because at 64 k-points the accumulate's
multiply-adds (`2·Q·nkpts·B` per lane, unchanged in count by fusion) are now
the pass; the block traffic K-10 removes was the smaller term there. Over
the whole plan the pure-PBE KRKS SCF on si 2×2×2 has gone 7.85 → **2.90 s**
(2.7×) and 4×4×4 45.8 → **11.4 s** (4.0×), with the k-symmetric driver at
2.31 / 10.66 s.

### 1.6 K-10v — the fused accumulate's k-loop as a vector (**bit-exact**)

§1.5's 4×4×4 rows said what was left: at 64 k-points the accumulate's
multiply-adds (`2·Q·nkpts·B` per lane) were the pass. `crates/pyscf-kernels/src/eval_gto.rs`
(`eval_ao_k_fused_kernel<N: Size>`), `crates/pyscf-kernels/src/pbc/eval_ao_k.rs`
(`AoKAccumulator::zeros_point_major`, `into_k_planes`).

The planes were k-major (`out[k·n + p]`), so consecutive k-points sat a
stride apart and the k-loop could not be a vector load. The fused kernel's
accumulator is now POINT-MAJOR (`out[p·nkpts + k]`): its k-loop runs over
`Vector<f64, N>` chunks — one multiply-add per image covers N k-points, the
phases `pr[m·nkpts + k..+N]` are a contiguous vector load, the value
broadcasts — with `N` the widest width the device likes for f64 that divides
`nkpts` (`line_size_for`; 8 k-points → 8, 6 → 2, 3 → 1). Per `(k, p)` the
sequence of additions is unchanged, so the planes are bit-identical; the
per-image `present` branch inside the innermost loop is gone too — an absent
image's value slots are zeroed and its multiply-add adds an exact `±0.0`,
which cannot change an accumulator that is never `-0.0` (it starts at `+0.0`,
and `+0.0 + -0.0 == +0.0`). Read-back gathers the point-major planes straight
into the per-k tensors the driver builds anyway (`into_k_planes`, one
strided pass per k, k-points in parallel through rayon — added to
`pyscf-kernels` for this), so the k-major intermediate the driver used to
copy from is gone as well. The k-major accumulator and every kernel that
writes it (K-08, K-09) are untouched; they refuse a point-major one.

**Bit-parity: EXACT** — `tests/eval_ao_image_batch.rs`, fused arms at the
cap / 7 / 1 on `[2,2,2]` (N = 8), plus a new test on `[1,2,3]` (6 k-points,
N = 2) and `[1,1,3]` (3, N = 1), both bases, both eval names, every real.

**Measured** — previous fused build (scalar k-loop) vs this one, load
2.2-4.0 (`scratchpad/k10v`, kept as `baselines/2026-09-08-k10v-*.json`):

| row | scalar k-loop | **vector k-loop** | ratio |
|---|---|---|---|
| gth-szv deriv 1, 2×2×2, cold pass | 1 632 ms | **1 038 ms** | **1.57×** |
| gth-szv deriv 1, **4×4×4** | 8 586 ms | **3 695 ms** | **2.32×** |
| gth-szv deriv 0, 4×4×4 | 2 138 ms | **1 248 ms** | 1.71× |
| gth-dzvp deriv 1, 2×2×2 | 6 511 ms | **3 474 ms** | **1.87×** |
| KRKS si 4×4×4 PBE `kernel()`, full BZ / ksymm | 13 975 / 12 840 ms | **8 089 / 6 896 ms** | **1.73× / 1.86×** |
| KRKS si 2×2×2 PBE `kernel()`, full BZ / ksymm | 2 898 / 2 313 ms (§1.5) | **2 831 / 2 196 ms** | 1.02× / 1.05× |

Energies identical to every printed digit on every row. Peak RSS unchanged
(the accumulator has the same size in either layout). The whole-plan
ledger for the pure-PBE KRKS SCF on si: 2×2×2 7.85 → **2.83 s** (2.8×),
4×4×4 45.8 → **8.09 s** (5.7×); k-symmetric 7.06 → 2.20 s and 43.1 →
6.90 s (6.2×).

## 4. The lever §1.5 then took — the reasoning that put it to the user

Written before D-PBC-32; kept as the record of why it was a decision and not
an item. With A-06 in, the cold pass at gth-szv deriv 1 is ~2.5 s: ~0.87 s AO
evaluation (lane scaffolding, §1.2) and ~1.6 s accumulate at batch 16, whose
traffic per image is one block read (`n · 8` B) plus `4·nkpts·n/B` of plane
read-modify-write. Batch 32 halves the second term (§1.4); the first is the
block itself — written by the evaluation, read by the accumulate. The one
change that removes it is **fusing the two**: a lane per `(g, shell)` that
evaluates the B images' values at its point and adds them straight into the
`nkpts` planes, image by image (still the same additions in the same order,
so still bit-exact; the shifted coordinate `r − L_m` computed in-kernel from
the unshifted grid, so no coordinate staging either). That is exactly the
"new periodic AO evaluator that folds the Bloch sum into the radial kernel"
PBC-MASTER-PLAN plan 10-04 forbids and plan-2 §1.2 lists as a non-goal. It is
recorded here with its measured motivation — after K-09 and A-06 it is the
only AO item left with a modelled gain above 1.3× — for the planning process
to decide, not taken unilaterally. The user took it (D-PBC-32, §1.5).

## 5. Not done, and why

* **v2 speed** — untouched (RULE M); v2 is still 0.02× the reference route.
* **M-13/M-14 same-box BEFORE arm** — still session 3's 3 962 MiB.
* **GATE A / GATE U** — not run (need `PYSCF_ORACLE_VENV=1`).

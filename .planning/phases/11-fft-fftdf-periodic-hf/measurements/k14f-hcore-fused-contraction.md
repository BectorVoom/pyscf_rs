# K-14f — `get_hcore`'s local contraction, fused on the device

Measured 2026-09-21. 16 cores, CubeCL CPU runtime (the default backend, ALG-03),
diamond `gth-dzvp` unless stated. Bit-identity gated by
`crates/pyscf-pbc-df/tests/hcore_fused.rs` and
`crates/pyscf-kernels/tests/pbc_local_vmat.rs` (12 tests).

## What changed

`get_hcore`'s local half is `v[k][p,q] = Σ_g conj(ao_k[p,g])·vR[g]·ao_k[q,g]` —
a reduction from `nkpts·nao·ngrids` complex AO values to `nkpts·nao²`. It used
to evaluate the AO table on the device, read ALL of it back, split it into per-k
host planes, and reduce those on the host. `pbc/local_vmat.rs` (K-14f) reduces
the device-resident accumulator in place instead: nothing of the table crosses
to the host and only the `nkpts·nao²` answer comes home.

`Fftdf::local_vmat` routes. `PYSCF_PBC_HCORE_FUSE` = `0` never / `1` always /
unset = `auto`, which fuses exactly when the table would NOT be admitted to the
AO cache. That predicate matters: fusing unconditionally would cost an SCF a
second cold AO pass, because today `get_hcore` leaves the table cached for
`get_j`/`get_k`.

## Peak RSS — `tests/hcore_peak_rss.rs`, ONE call per process

Per-process isolation is required: CubeCL pools device buffers per
process-global client, so a second arm in the same process inherits the first
one's pool and both report the same `VmHWM`. `VmHWM` is reset with
`/proc/self/clear_refs`. diamond `gth-szv` (nao 8).

| nk, mesh | AO table | host peak Δ | fused peak Δ | saved |
|---|---|---|---|---|
| 2³, 31³ | 29.1 MiB | 199.6 MiB | 169.0 MiB | 30.6 MiB (−15 %) |
| 2³, 41³ | 67.3 MiB | 272.5 MiB | 224.6 MiB | 47.9 MiB (−18 %) |
| 3³, 41³ | 227.2 MiB | 800.7 MiB | 588.3 MiB | 212.4 MiB (−27 %) |

The saving is ~1× the AO table and its share of the call grows with size.

**What it does NOT remove.** `ao_table_only` (the AO evaluation and read-back
alone, no contraction) peaks at 791.2 MiB on the 3³/41³ row — 3.5× the table,
and the fused route still pays 588.3 of it. The AO EVALUATION stage, not the
read-back, sets the peak: the accumulator's two planes, the `vec![0.0; nkpts·n]`
they are uploaded from, and the K-09 image batch. Driving the grid in blocks
would cut that too, at the cost of re-running the lattice-image loop per block.
That is the next lever, and it is NOT this item.

## Time — `krks_profile hcore --arm`, diamond gth-dzvp 3³, mesh 41³

nao 26, ngrids 68 921, nkpts 27, AO table 738.3 MiB. `--reps 2`, fresh `Fftdf`
per rep (so no rep hits the AO cache), warm rep quoted.

| arm | ms | peak RSS |
|---|---|---|
| host | 6 721.7 | 1 926.4 MiB |
| fused | 5 909.7 | 1 662.9 MiB |

1.14× faster and −13.7 % peak, bit-identical.

## The point-major trap (the whole first attempt)

`AoKAccumulator` has two layouts and the fast one for the AO evaluation is the
slow one for this reduction. The K-10v fused AO path accumulates POINT-MAJOR
(`plane[e·nkpts + k]`, so its k-loop vectorises), which puts successive `g` of
one AO `nkpts` apart — a lane walking `g` then touches a new cache line on every
load. Contracting those planes in place:

| AO layout (`PYSCF_PBC_AO_FUSE`) | host | fused (in place) |
|---|---|---|
| point-major (`1`, the default) | 6 731.9 ms | 12 271.8 ms |
| k-major (`0`) | 13 886.1 ms | 13 130.4 ms |

The kernel was fine — the stride was the entire difference (fused beat host on
k-major planes and lost 1.8× on point-major ones). Fixed by gathering one k at a
time into a contiguous `n`-element scratch (`1/nkpts` of the table, pure data
movement) and contracting with `stride_e = 1`. That is the 5 909.7 ms above.

**Two traps for anyone re-measuring this:**

1. A peak RSS taken after several reps in one process is the CubeCL pool's
   high-water mark, identical for both routes and meaningless. `krks_profile
   hcore` now reads it after the FIRST rep only; the probe takes one call per
   process.
2. Letting the host arm share an `Fftdf` across reps makes reps 2+ hit the AO
   cache, so the arm reports a cached reduction (31 ms) as if it were the whole
   call (2 078 ms).

## B-00 — measured term split

Measured 2026-09-21. Same shape as §2.2's arithmetic (diamond `gth-szv`,
nao 8, `PROBE_NK=3`, `PROBE_MESH=41`: `nkpts` 27, `ngrids` 68 921, one AO
table 227.2 MiB). One call per process (`tests/hcore_peak_rss.rs`, C4):

| probe | peak Δ | × table |
|---|---|---|
| `ao_table_only` | 789.5 MiB | 3.48× |
| `host_route` | 800.4 MiB | 3.52× |
| `fused_route` | 589.5 MiB | 2.60× |
| `accumulator_only` (two device planes + host zeros `Vec`) | 462.9 MiB | 2.04× |
| `image_batch_only` (`capacity` 32 slots of `n = nao·ngrids`) | 8.9 MiB | 0.04× |

Against §2.2's predictions (planes 227.2 + zeros 113.6 = 340.8 MiB;
batch `32·n·8` = 134.4 MiB; total 475.2 MiB):

* `accumulator_only` measured 462.9 MiB against 340.8 MiB predicted (+122.1 MiB).
* `image_batch_only` measured 8.9 MiB against 134.4 MiB predicted (−125.5 MiB).
* The SUM measured 471.8 MiB against 475.2 MiB predicted (−3.4 MiB, <1%).

The batch shortfall is almost certainly lazy page faulting, not absence:
`AoImageBatch::new` reserves its slots with `client.empty`, which on the CPU
runtime maps address space without faulting pages, and this probe never
writes a slot — so the 134.4 MiB stays virtual and out of RSS. In a real
call every slot is evaluated into before the accumulate, faulting the pages
and making the batch resident. Conversely the accumulator's uploads write
every page (plus transient staging), which overshoots the naive
planes + zeros sum. Per-term RSS attribution therefore needs a probe that
writes the batch; the totals already account for every predicted byte.

§2.2's arithmetic DID NOT HOLD: accumulator_only measured 462.9 MiB against 340.8 MiB predicted.

§2.2's arithmetic DID NOT HOLD: image_batch_only measured 8.9 MiB against 134.4 MiB predicted.

## B-01 — post-change C4 (3³/41³, table 227.2 MiB)

`AoKAccumulator::zeros` no longer builds the host `vec![0.0; nkpts·n]`; the
two planes are `client.empty` + device `fill_zero` (`crates/pyscf-kernels/src/pbc/fill.rs`).

| probe | B-00 | post-B-01 | Δ |
|---|---|---|---|
| `ao_table_only` | 789.5 MiB | 788.9 MiB | −0.6 (noise) |
| `host_route` | 800.4 MiB | 797.5 MiB | −2.9 (noise) |
| `fused_route` | 589.5 MiB | 588.1 MiB | −1.4 (noise) |
| `accumulator_only` | 462.9 MiB | 9.4 MiB | −453.5 |
| `image_batch_only` | 8.9 MiB | 8.9 MiB | 0 |

Two readings, both honest:

1. `accumulator_only` dropped 453.5 MiB, ~4× the predicted `nkpts·n·8` =
   113.6 MiB. The excess is CubeCL laziness, not extra saving: the probe never
   reads back, so pre-B-01 the `upload` path eagerly staged host bytes while
   post-B-01 the `empty` planes + fill kernel never execute or fault — the
   probe measures reservations, not executed work.
2. Neither end-to-end route moved. The transient zeros `Vec` is dropped right
   after `zeros()` returns, so it never sets the call's `VmHWM` at this shape —
   the watermark is set later in the image loop (planes + batch + eval
   staging + pool slack). The 113.6 MiB transient is really gone (no
   `vec![0.0f64` remains in `eval_ao_k.rs`), but it was never the watermark.
   B-04 must use route peaks, not probe deltas, as its `mem_saving` input.

## B-02 — post-change C4 (3³/41³, table 227.2 MiB)

The K-09 batch budget is now `min(256 MiB, accumulator/2)`. At this shape the
accumulator is 238 MiB, so the budget drops 256 → 119 MiB and the driver
capacity 32 → 27. (`image_batch_only` still reports 8.7 MiB — the probe
constructs capacity 32 directly and never writes a slot, so it is insensitive
to the driver formula by construction; see the B-00 note on lazy faulting.)

| probe | pre-B-02 | post-B-02 | Δ |
|---|---|---|---|
| `ao_table_only` | 788.9 MiB | 790.7 MiB | noise |
| `host_route` | 797.5 MiB | 799.7 MiB | noise |
| `fused_route` | 588.1 MiB | 585.8 MiB | noise |
| `image_batch_only` | 8.9 MiB | 8.7 MiB | 0 |

No route peak moved: at 32 → 27 images per launch the batch was never the
watermark at this shape. `PYSCF_PBC_AO_IMAGE_BATCH=1` still forces capacity 1
(`pbc_eval_ao_k` 8/8 and `eval_ao_image_batch` 3/3 pass with the override).

## B-03e — C4 by block size (3³/41³, table 227.2 MiB)

`PYSCF_PBC_AO_GRID_BLOCK` unset vs `8192` vs `1024`, one process per probe:

| probe | whole-grid | BLK=8192 | BLK=1024 |
|---|---|---|---|
| `ao_table_only` | 789.4 MiB | 786.7 MiB | 788.7 MiB |
| `host_route` | 797.8 MiB | 796.5 MiB | 799.6 MiB |
| `fused_route` | 587.6 MiB (2.59×) | 388.5 MiB (1.71×) | 146.5 MiB (0.65×) |
| `accumulator_only` | 9.5 MiB | 9.3 MiB | 9.3 MiB |
| `image_batch_only` | 9.4 MiB | 9.1 MiB | 9.1 MiB |

The host route (whole-table read-back) does not move, as designed — blocking
only changes the fused contraction path. The fused route drops −34% at 9
blocks and −75% at 68 blocks, below one AO table. The BLK=8192 saving
(199.1 MiB) matches §2.3's model (accumulator 227.2 → 27.0 MiB = 200.2 MiB)
to within a megabyte.

## B-03e — C5 by block size (diamond gth-dzvp 3³, mesh 41³)

Table 738.3 MiB (nao 26). Warm rep of 2, one process per arm:

| BLK | host ms | host peak | fused ms | fused peak |
|---|---|---|---|---|
| whole-grid | 7298.7 | 1933.9 MiB | 6215.8 | 1231.4 MiB |
| 8192 | 7476.7 | 1932.7 MiB | 6355.1 | 552.2 MiB |
| 1024 | 7356.1 | 1930.1 MiB | 6926.5 | 285.4 MiB |

The host arm is untouched by the switch (identical ms and peak across BLKs —
it never enters the blocked path). Fused `time_cost`: +2.2% at 8192, +11.4%
at 1024; fused peak: −55% / −77%. (Load average 9–11 during the run — another
session was building on the box — so the ms figures carry noise; B-04
re-measures.)

B-03e RISK arm (point-major gather scratch per block), BLK=1024, fused route:

| `PYSCF_PBC_AO_FUSE` | fused ms | fused peak |
|---|---|---|
| `1` (point-major + per-block gather, the default) | 6841.4 | 283.1 MiB |
| `0` (k-major, no gather) | 9632.7 | 230.4 MiB |

The gather does not set `BLK` against the default: FUSE=1 stays 1.4× faster
blocked, because the K-10 fused evaluation it enables outweighs the
per-block gather. FUSE=0 peaks lower (no K-09 batch, k-major planes) but is
not the default and not the B-04 input.

## B-04 — verdict (measured 2026-09-21/22)

C4 peaks (gth-szv `fused_route` Δ, one call per process) and C5 times (gth-dzvp
`fused` arm, warm rep of 2), whole-grid reference vs BLK arms:

| shape | table | peak whole | peak 8192 | peak 1024 | time whole | time 8192 | time 1024 |
|---|---|---|---|---|---|---|---|
| 2³, 31³ | 29.1 MiB | 174.2 | 149.4 | 134.6 | 1131.2 ms | 1189.9 ms | 1133.0 ms |
| 2³, 41³ | 67.3 MiB | 211.9 | 150.7 | 137.9 | 2485.6 ms | 2549.7 ms | 2690.1 ms |
| 3³, 41³ | 227.2 MiB | 585.3 | 388.8 | 145.1 | 6215.8 ms | 6355.1 ms | 6926.5 ms |

`mem_saving = 1 − peak(BLK)/peak(whole)`, `time_cost = time(BLK)/time(whole) − 1`:

| shape | 8192 saving | 8192 cost | 1024 saving | 1024 cost |
|---|---|---|---|---|
| 2³, 31³ | 14.2% | +5.2% | 22.7% | +0.2% |
| 2³, 41³ | 28.9% | +2.6% | 34.9% | +8.2% |
| 3³, 41³ | 33.6% | +2.2% | 75.2% | +11.4% |

Per-BLK application of the decision table:

* **BLK=8192: row 1 at all three shapes** (saving > 10%, cost ≤ 10%).
  **Default ON.** The row asks for BLK derived from device cache properties;
  CubeCL 0.10's `DeviceProperties` exposes load width, plane sizes, cube
  counts, SM/core counts and memory alignment — no cache size — so a
  cache-derived BLK is not implementable. The default is the measured 8192
  (`AO_GRID_BLOCK_DEFAULT`), recorded here as measured, not derived.
  `PYSCF_PBC_AO_GRID_BLOCK=0` remains the whole-grid opt-out.
* **BLK=1024 (and smaller): row 1 at small shapes, row 2 at the flagship**
  (+11.4% time for 75.2% memory). **Opt-in via the switch; default stays
  8192.** Exchange rate at the flagship: −440 MiB peak per +711 ms (≈1.6 ms
  per MiB saved).

B-02's provisional batch fraction, resolved by measurement: peak and time are
flat across image-batch capacities 2–32 at the flagship (peaks 582–593 MiB,
times 6023–6416 ms, noise), so `AO_IMAGE_BATCH_ACC_NUM/DEN = 1/2` stands as a
safe ceiling that preserves the K-09 batching — not a tuned optimum. The only
cliff is capacity 1 (the legacy per-image fallback, never produced by the
formula at real shapes): −41% peak (342.3 MiB) for 8.6× time (52 045 ms).
Do not lower the fraction toward 1 chasing memory.

## B-04 follow-up — BLK is autotuned, not fixed (supersedes the fixed default)

The table above shows the winning `BLK` is shape-dependent: `1024` was fastest
at 2³/31³, `8192` at 2³/41³ and 3³/41³. A single constant cannot be right
everywhere, so the fixed default is replaced by cubecl autotuning
(`crates/pyscf-algebra/src/autotune.rs`, ALG-09).

**What is tuned.** The candidates are `GRID_BLOCK_CANDIDATES = [8192, 1024]` —
the two arms B-04 measured. The tuner benchmarks every memory-viable one and
keeps the fastest, per shape and per device, persisting the choice to cubecl's
on-disk autotune cache (`target/`, `CacheConfig::Target` default; verified to
survive across processes).

**The memory constraint.** Autotune optimizes *time*, but blocking exists for
*memory*, so a candidate is pruned (a negative `TuneGroup` intra-priority, the
manual's early-stop) unless its block accumulator is at least
`GRID_BLOCK_SHRINK = 4`× smaller than the whole-grid accumulator. The smallest
candidate is always viable, which keeps the tuner's plan non-empty. Pruned
candidates are never compiled, never benchmarked, never selected.

**Cost.** `tune_benchmark` runs 3 warmup + 10 measured executions per viable
candidate on the real inputs, so a cold key costs `13 × viable` full
evaluations; the winner is cached and later calls run once. Only the larger
grids have both candidates viable: at 2³/31³ the shrink floor leaves `{1024}`
(single candidate, no benchmarking), at 2³/41³ and 3³/41³ it leaves
`{8192, 1024}` (26 evaluations, once). `PYSCF_PBC_AO_GRID_BLOCK=<multiple of
128>` skips the tuning entirely when that one-time cost is unwanted.

**Bit-parity.** Every `BLK` is bit-identical, gated by
`crates/pyscf-pbc-df/tests/hcore_block.rs`; the autotuned path is compared
against the whole-grid reference there too. The tuner's own contract (which
candidates run, which are pruned, warm-cache single execution) is gated by
`crates/pyscf-algebra/tests/autotune.rs`.

**Switch.** `PYSCF_PBC_AO_GRID_BLOCK`: unset or `auto` = autotune (the
default); `0` = whole grid; a multiple of 128 = that fixed `BLK`, no
autotuning. All bit-identical. The fixed values remain the escape hatch when
the one-time tuning cost is unwanted.

## B-04 follow-up — autotune verified end to end (2026-09-22, fresh binary)

Switch unset (= autotune), one process per arm:

| probe | cold | warm | fixed-8192 reference |
|---|---|---|---|
| C4 szv 3³/41³ `fused_route` Δ | 416.9 MiB | 397.6 MiB | 388.8 MiB |
| C5 dzvp 3³/41³ fused ms / peak | rep1 175136 ms, peak 609.2 MiB | 6267.5 ms / 550.6 MiB | 6355.1 ms / 552.2 MiB |

* The cold rep1 (175 s ≈ 26 evaluations × ~6.5 s) is the one-time tuning pass;
  its peak (609 MiB) stays bounded by one block, as designed.
* The warm numbers equal fixed-8192 to noise. The on-disk cache confirms why:
  flagship dzvp key `{27, 26, 1, 68921, 713}` committed winner index 0 =
  `blk-8192` (median 6.17 s vs `blk-1024` 6.73 s); the C4 szv key committed
  `blk-8192` likewise.
* Bit-parity of the autotuned path is gated against the whole-grid reference
  in `crates/pyscf-pbc-df/tests/hcore_block.rs`
  (`autotuned_get_pp_is_bit_identical_to_whole_grid`).

Operational note: cubecl's `CacheConfig::Target` resolves the cache root by
walking up to the first `Cargo.toml`, so each calling crate keeps its own
cache (`crates/pyscf-pbc-df/target/autotune/…` for tests,
`./target/autotune/…` for workspace-root binaries). First call per binary per
shape pays the tuning pass; every later call is a single execution.

## Review correction (2026-09-22) — the default is fixed 8192, autotune is opt-in

The "autotune by default" follow-up above did not re-apply B-04's decision
table, and fails it on a cold key: 26 full evaluations (175 s against 6.2 s at
gth-dzvp 3³/41³), paid again for every new `(nkpts, nao, ngrids, nimgs)`, and at
both shapes where it had a choice it picked 8192 — the row-1 verdict. So:
`PYSCF_PBC_AO_GRID_BLOCK` unset = fixed `AO_GRID_BLOCK_DEFAULT = 8192`;
`0` = whole grid; a multiple of 128 = that `BLK`. (An interim `auto` opt-in
was then removed with the tuner — see below.)
Gated by `default_get_pp_is_bit_identical_to_whole_grid` (33³, five blocks).

Also fixed in the same pass: `eval_ao_kpts_local_vmat_blocked(.., blk = 0)`
looped forever and non-multiples of 128 panicked; both now return an error
(`a_bad_block_size_is_an_error_not_a_hang`).

## Autotune REMOVED (2026-09-22)

`crates/pyscf-algebra/src/autotune.rs` (ALG-09), its tests, the
`autotune-checks` feature and the `serde` dependency it needed are deleted, and
`PYSCF_PBC_AO_GRID_BLOCK=auto` is no longer accepted (it now warns and uses one
block, like any other unrecognised value). The two "B-04 follow-up" sections
above describe code that no longer exists; they are kept as the record of why.
The reasons, all from the measurements above: a cold key cost 26 full
evaluations (175 s against 6.2 s), every new shape paid it again, and at both
shapes where the tuner had a choice it chose 8192 — the fixed default. It cost
time and bought nothing measured.

## Probe fix — `accumulator_only` now measures EXECUTED planes

The B-00/B-01 `accumulator_only` numbers above (462.9 → 9.4 MiB) measured
reservations, not memory: building the accumulator only queues its fill
kernels, and `client.empty` faults no pages until something runs. The probe now
forces execution by contracting the planes with `local_vmat_resident`, which
runs the queued fills and reads every page while bringing home only
`nkpts · nao²` numbers. Re-measured at 3³/41³ (table 227.2 MiB):

| probe | before (reservation) | now (executed) |
|---|---|---|
| `accumulator_only` | 9.4 MiB | 286.2 MiB (1.26×) |

286.2 MiB against the predicted 227.2 MiB of planes; the remainder is pool and
JIT overhead of the first launches. `image_batch_only` is deleted rather than
fixed: its slots are written only by the AO evaluator, so no probe can make them
resident without running the evaluation `ao_table_only` already measures. Treat
the `image_batch_only` rows above as void.

## B-02 correction (2026-09-23) — the batch budget collapsed at small `nkpts`

B-02's accumulator-relative budget reduces, algebraically, to `nkpts`: the
driver passes `accumulator_bytes = 2·nkpts·n·8` and `block_bytes = n·8`, so
`accumulator/2 / block_bytes == nkpts`. The term therefore binds hardest where
memory pressure is LOWEST, and at gamma it asked for capacity **1** — the
pre-K-09 per-image fallback. B-04's note that capacity 1 is "never produced by
the formula at real shapes" was wrong; gamma is a real shape.

No bit-identity gate could catch it (batching is bit-exact at any capacity), so
it showed only as wall clock. Measured on the `hcore_block`
`blocked_local_vmat_is_bit_identical` body (gamma + 2×2×2, small k throughout):

| batch capacity | before the floor | after the floor |
|---|---|---|
| default (computed) | 1.80 s | **0.93 s** |
| pinned 32 | 0.77 s | 0.76 s |
| pinned 1 (fallback) | — | 2.04 s |

Fix: `AO_IMAGE_BATCH_MIN = 8` floors the capacity, itself capped by the
absolute 256 MiB budget so the floor can never enlarge the batch past it.
Justified by B-04's own sweep — peak and time flat across capacities 2–32 at
the flagship, cliff only at 1.

Residual: the default (floor 8) is still ~1.2× slower than a pinned 32 at
gamma, because the accumulator term still caps capacity below the absolute
budget there. Raising the floor to 32 would recover it and would make the
accumulator term inert — which is arguably what the measurements say it should
be. Left at 8 as the conservative choice; revisit with a peak measurement at
gamma before raising.

Gate: `crates/pyscf-pbc-gto/tests/ao_image_batch_floor.rs` — a deterministic
capacity assertion over `(nkpts, nao, ngrids)` shapes (the one to trust), plus
a same-process wall-clock ratio against a pinned capacity 1.

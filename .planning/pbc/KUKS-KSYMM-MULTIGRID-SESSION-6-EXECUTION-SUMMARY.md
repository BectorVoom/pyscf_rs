# KRKS + k-symmetry + multigrid — session 6 execution record

**Plan:** [`KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md`](./KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md);
session 5's record: [`KUKS-KSYMM-MULTIGRID-SESSION-5-EXECUTION-SUMMARY.md`](./KUKS-KSYMM-MULTIGRID-SESSION-5-EXECUTION-SUMMARY.md),
whose §5 left "v2 speed — untouched (RULE M)" and "the host copies of the
batch geometry" as the open multigrid rows.
**Date:** 2026-09-08 (same day, same box, same rules).
**Ask (user, verbatim):** "optimising memory efficiency and speed KRKS and
k-point symmetry + multigrid gpu kernel". RULE M (v2 is a memory-shape item,
not a promised win) is therefore lifted for this session: the v2 pair
kernels ARE the multigrid GPU kernels, and their speed is in scope.
**Machine / load:** 16 cores, 30 GiB, CubeCL CPU runtime (no f64 GPU here,
`rocm-igpu-no-f64`); every GPU statement below is REASONED and marked so.
The 1-minute load at launch is quoted beside every number.
**Manuals read before touching a kernel (RULE 5):** `profiling_tools.md`,
`16_profiling_and_bottleneck_identification.md` (§2.4 warm-up / many reps /
report the timing method; §2.6 the CPU runtime's one-thread-per-unit model),
`06_vectorization.md` + `Cubecl_dynamic_vectorization.md` (`Vector<F, N>`,
the width as a launch argument, `ArrayArg` in scalar length),
`07_memory_coalescing.md` (§3 SoA, §4 put the inner loop on the unit stride,
§5 narrow the element type), `Cubecl_loop_control.md` (range loops, no
`continue`), `Cubecl_conditionals.md` (`if` as a statement). Read from the
cubecl 0.10 sources: `Vector<E, N>` implements `CubeIndex`/`CubeIndexMut`
(element read/write, `frontend/operation/assignation.rs`), `N::value()` is
a comptime `usize` (`frontend/element/base.rs`), `#[unroll] for i in
0..N::value()` is the idiom cubecl's own `runtime_tests/unroll.rs` uses;
`cube_math::double::exp` is SCALAR and glibc-bit-exact
(`cube-math/src/double/exp.rs`), so a vector kernel assembles N scalar calls.

---

## 0. What this session did

| item | what | result |
|---|---|---|
| **M-15** term sets | the per-concatenated-slot `slot_global` table (77.8 M `u32`, 311 MB at `25³` level 3) replaced by one set index per distinct instance and the level's few thousand `(set_off, set_pow)` entries; the slot loops read powers and coefficients SEQUENTIALLY | landed, **bit-exact**; level-3 resident geometry 622 → **259 MB** |
| **M-16** device fold | the reverse read-back (`nslots · 8` = 622 MB per call at level 3) and its host loop replaced by `mg_fold_kernel` onto a level-resident `kint` (`nkslots · 8` = 11 MB read once per level) | landed, **bit-exact** (same additions, same order, chunk after chunk) |
| **M-17** vector forward | `mg_rho_kernel<N>`: one lane per N adjacent (padded) points; N scalar `exp`s per instance, everything else `Vector<f64, N>` | landed, **bit-exact** at N = 1, 2, 8 and the device width; **3.2×** on the level-3 forward (§3) |
| **M-18** host release | the chunk's host geometry is TAKEN on first upload (`PYSCF_MG_PAIR_KEEP_HOST=1` keeps it) | landed, bit-neutral; RSS §3 |
| **M-19** vector reverse | `mg_integrate_vec_kernel<N>`: one lane per N consecutive occurrences of a block, predicated power products (`poly *= dx·m + (1−m)`), hoisted masks, N scalar `exp`s per point | landed, **bit-exact**; **1.54×** over the term-set scalar reverse, 1.75× over session 5's (§3); `PYSCF_MG_PAIR_REVERSE=scalar` is the A/B arm |
| instrument | `tests/mg_pair_bench.rs` (`#[ignore]`): per-level geometry statistics (sets, same-set run lengths) and the isolated warm forward/reverse wall; `PYSCF_MG_PAIR_EXP=exact|fast|none` kill-switch arm; `PYSCF_MG_PAIR_LINE=1|2|4|8` width pin | landed |
| GATE S | S-03 (`PYSCF_PBC_KSYMM_RHO=symmetrize`) measured at the gate mesh, §5 | measured |

Everything above went through `tests/multigrid_batch.rs`'s `to_bits()`
comparison against the per-block streaming route (§4), which is the whole
of the bit-exactness claim: no tolerance anywhere.

---

## 1. Where v2's time and memory were (BEFORE)

Session 5's idle-box numbers (`baselines/2026-09-08-multigrid-v2-si-mesh25.json`,
load 2.8), Si gth-szv LDA at `25³`, warm `get_veff` **9 722 ms**, peak RSS
**2 411 MiB** (after M-14):

| level | forward | reverse | occurrences | distinct | concatenated slots | kernel slots |
|---|---|---|---|---|---|---|
| 1 (`11³`) | 11 ms | 11 ms | 0.17 M | 27 k | 0.17 M | 27 k |
| 2 (`15³`) | 1 160 ms | 1 079 ms | 3.7 M | 0.70 M | 25.1 M | 1.27 M |
| 3 (`25³`) | 3 928 ms | 3 529 ms | 13.8 M | 2.28 M | 77.8 M | 1.38 M |

Reading the kernels against those counts (`multigrid_pair.rs` before this
session): the forward lane (one grid point) walked its block's ~110 k
occurrences and, per occurrence, ~5.6 slots through THREE dependent gathers
(`slot_global[s]` → `kslot_pow[k]`, `kcoef[k]`), i.e. 9.7 G random
gathers per level-3 sweep; the reverse lane (one occurrence) did the same
per point, then wrote one value per concatenated slot, all 622 MB of which
were read back and folded on the host every call. The `exp` count
(`occurrences × points` = 1.73 G per direction) is set by the block-level
reach test and is untouched here — it is the next lever (§6).

The geometry statistics the new instrument prints (`mg_pair_bench`, level 3):
14 304 term sets over 71 832 set slots serve 249 k level instances and
13.8 M occurrences — the per-slot table was 1 000× redundant — and 93 % of
occurrences sit in same-set runs of ≥ 4 inside a block (85 % in runs ≥ 8),
which is why a vector-over-occurrences reverse kernel wastes almost nothing
on ragged tails.

---

## 2. M-15 / M-16 / M-17 / M-18 / M-19 — the pair kernels re-laid

`crates/pyscf-kernels/src/multigrid_pair.rs` (the M-03 section, rewritten),
`crates/pyscf-pbc-dft/src/multigrid/pair.rs` (`PairLevelTable::{instance_set,
instance_kslot0, set_off, set_pow, set_term}`, `BatchedLevel`,
`build_batch_geometry`, `level_scratch`, the four drivers),
`tests/multigrid_batch.rs`, `tests/mg_pair_bench.rs`.

**M-15.** `build_pair_level_table` already pushed one `terms_here` sequence
per `(pair, L)` and copied it into every wrap image; the sequence is now a
SET (`set_off`/`set_pow`/`set_term`), each instance carries its set index,
and `instance_kslot0` records where the instance's kernel slots start. A
chunk stores sets once (a few KB), one set index and one kernel-slot base
per distinct instance, and — for the reverse output — one prefix per
occurrence (`occ_slot0`, the old `inst_slot0`). `slot_global` is gone. The
forward per-call coefficient is per SET slot (`set_coef[so..so+n] =
term_coef[set_term[..]]`, 72 k reals at level 3 instead of 1.38 M), read
sequentially by the kernel. The reverse fold's target `k = instance_kslot0
+ j` is exactly the old `slot_global[s]` (a block's reach list is in kernel-
slot order and an instance's kernel slots are contiguous), so the fold
sequence is unchanged.

**M-16.** `mg_fold_kernel`: one lane per `(distinct instance, monomial)`,
`MAX_SLOTS_PER_INSTANCE` lanes per instance (the surplus idle); the lane
starts from the running `kint[k]`, adds the chunk's outputs for that
instance in OCCURRENCE order (`uocc_off`/`uocc`, a counting sort built with
the chunk), and stores. `kint` lives in the level's `PairOutScratch`
(`zero_kint` / `read_kint`), zeroed by a `mg_zero_kernel` launch before the
first chunk and read once after the last; the two-spin path has `kint_b`
beside it, shared by every chunk of the level (a per-chunk `OnceLock` would
have folded each chunk into its own buffer — caught in review, not by a
gate, since the KUKS gate ran after the fix).

**M-17.** Blocks are padded to a multiple of `POINT_PAD = 8` points (pads
repeat the block's last real point; `point_global[pad] = PAD_POINT`), so
any width in {1, 2, 4, 8} divides every block. `mg_rho_kernel<N>` reads
`Array<Vector<f64, N>>` coordinates, broadcasts the instance's centre and
`eta`, forms `arg = 0 − eta·r²` elementwise and `e[j] = exp(arg[j])` per
element, then the same `while` power loops over vector `poly` and `acc +=
coef · poly · e`. Each point sees the scalar kernel's operations in the
scalar kernel's order. The reverse kernel stops at `block_point_end[b]`, so
pads never enter an integral. The width is `line_size_for::<f64>(client,
POINT_PAD)` (8 on this runtime) or the `PYSCF_MG_PAIR_LINE` pin.

**M-19.** `mg_integrate_vec_kernel<N>`: a group is N consecutive
occurrences of one block (`grp_occ0`, built per chunk for the device's
width at upload); a ragged tail duplicates the block's last occurrence into
its unused elements, whose results are never stored. Per point: `dx = x −
cx` (vector), N scalar `exp`s, `we = w · e`, and per slot the predicated
products `poly *= dx·m + (1 − m)` with `m ∈ {0, 1}` from the packed powers
(`m0 = (ix & 1) | (ix >> 1)`, `m1 = ix >> 1`, hoisted out of the point loop
as 6 mask vectors per slot). `dx·1 + 0 = dx` and `dx·0 + 1 = 1` exactly, so
a lane with power `p` performs the scalar kernel's `p` multiplications in
the scalar kernel's order; the one difference — a `−0.0` where the scalar
path had `+0.0·dx` with `dx = −0.0` — is a sign of zero no accumulator can
observe (an accumulator starting at `+0.0` never becomes `−0.0`).
`validate_batch` refuses a set with a power above 2. Locals are ~5 KB per
lane at N = 8, so the launch goes through `launch_1d_chunked`
(`cubecl-cpu-local-arrays-cost-stack-per-iteration`).

**M-18.** `BatchedLevel.batch` is a `Mutex<Option<PairSlotBatch>>` taken by
`resident_batch` on first upload; the chunk's shape (`npoints`, `ninstances`,
`nuinstances`, `nslots`, `geometry_bytes`) is recorded beside it for the
instrument, and `BatchedLevel::host_batch` rebuilds the geometry from the
level table when a caller (a test, or a client of another backend) needs
it again. The bytes model that cuts chunks (`batch_bytes`) now counts the
resident geometry plus the reverse output; level 3 cuts into 4 chunks
instead of 13.

**GPU (REASONED, UNVERIFIED here):** the forward kernel's lanes shrink by N
(2 000 lanes for a `25³` level at N = 8), which is the wrong direction for a
discrete GPU's occupancy — a GPU client should pin `PYSCF_MG_PAIR_LINE=1`
(or 2) until measured, and the item that would restore occupancy there
(splitting each point's instance range across lanes) changes the summation
order and is NOT taken. The reverse groups (1.7 M lanes at N = 8) and the
fold (8 M lanes) are GPU-shaped; the sequential set walk is a broadcast
load within a plane of same-block lanes; M-16 removes the 622 MB per-call
device→host transfer, which on PCIe was the reverse direction's floor.

---

## 3. Measured — isolated kernels (`mg_pair_bench`, Si gth-szv `25³`, warm min over 3, `target/gate` no-LTO)

Same binary, arms by environment (`scratchpad/s6/ab/bench-*-25.log`); the
load column is the 1-minute load at the arm's start (each arm drives 16
threads, so a series' own predecessors raise it):

| level | arm | forward | reverse | load |
|---|---|---|---|---|
| 3 | **default** (N = 8 forward, N = 8 reverse groups) | **1 038 ms** | **2 017 ms** | 4.8 |
| 3 | `PYSCF_MG_PAIR_REVERSE=scalar` (M-15 term-set scalar reverse) | 1 064 | 3 097 | 5.6 |
| 3 | `PYSCF_MG_PAIR_LINE=1` (scalar forward, 1-wide groups) | 3 341 | 5 251 | 8.6 |
| 3 | `PYSCF_MG_PAIR_LINE=4` | 1 382 | 2 312 | 12.0 |
| 3 | `PYSCF_MG_PAIR_EXP=none` (kill switch: the `exp` share) | 455 | 1 034 | 11.7 |
| 3 | `EXP=none` + `REVERSE=scalar` | 457 | 2 047 | 11.3 |
| 2 | default | 303 | 619 | — |
| 2 | `REVERSE=scalar` | 298 | 950 | — |
| 2 | `LINE=1` | 1 006 | 1 624 | — |
| 2 | `EXP=none` | 135 | 319 | — |

Against session 5's idle rows (§1): level 3 forward 3 928 → **1 038 ms
(3.8×)**, reverse 3 529 → **2 017 ms (1.75×)**, the level 7 457 → 3 055 ms
(**2.4×**); level 2 2 239 → 922 ms (2.4×). The `LINE=1` arm reproduces the
old forward (3.3 s), so the forward gain is the vector lane (M-17 with M-15
under it), not the term sets alone; the scalar-reverse arm puts M-15 + M-16
at 1.14× on the reverse and M-19 at a further 1.54×. Half of what remains
is the `exp` (forward 56 %, reverse 49 %): 1.73 G scalar `cube_math` calls
per direction, at ~1.7 ns each across 16 threads. Checksums are identical
across the `LINE` / `REVERSE` arms (and differ, as they must, under
`EXP=none`).

### 3.1 Whole SCF — `krks_profile multigrid --driver krks --numint v2 --mesh 25,25,25`, BEFORE vs AFTER release binaries

Thin-LTO `--release` binaries of the tree before and after this session
(`scratchpad/s6/krks_profile.{before,after}`), run back to back on the
idle box (`baselines/2026-09-08-s6-multigrid-v2-si-mesh25-{before,after}.json`;
the AFTER's load 6.3 is the BEFORE run's own tail):

| | BEFORE (load 1.05) | **AFTER** (load 6.3) | ratio |
|---|---|---|---|
| `kernel()` to convergence | 30 685 ms | **14 981 ms** | **2.05×** |
| warm `get_veff` | 9 501 ms | **4 038 ms** | **2.35×** |
| level 3 forward / reverse | 3 860 / 3 428 ms | **1 069 / 2 048 ms** | 3.6× / 1.67× |
| level 2 forward / reverse | 1 137 / 1 051 ms | **305 / 596 ms** | 3.7× / 1.76× |
| level 3 launches per direction | 13 | **4** | |
| level 3 per-call transfer after (RULE T) | 765.5 MB | **13.6 MB** | 56× |
| level 3 resident distinct instances (chunks are larger, so fewer duplicates) | 2 278 963 | 818 323 | |
| peak RSS | 2 551 MiB | **1 640 MiB** | **−36 %** |
| `e_tot` | −7.160554062714283 | **−7.160554062714283** | identical |

Over the plan: v2's warm `get_veff` at `25³` 11 621 → 4 038 ms (2.9×) and
its peak RSS 4 390 → 1 640 MiB (−63 %) since session 3 §2, all of it
bit-exact against the per-block route, the SCF energy unchanged to every
printed digit across four sessions. v2 is still 425× v1's warm `get_veff`
(9.5 ms, whose level values are cached across the SCF); the ratio is the
work model — `occurrences × points` exponentials per direction, §6 — not
the kernels' shape any more.

---

## 4. Gates

No-LTO `target/gate` binaries (`CARGO_PROFILE_RELEASE_LTO=false`, the
spelling the directory was built with — `=off` re-fingerprints the profile
and rebuilds the libxc tree, memory `gate-target-dir-lto-spelling`), one
`systemd-run` scope (`MemoryMax=20G`) per binary, `scratchpad/s6/gates/*.log`.

| gate | result |
|---|---|
| `pyscf-pbc-dft` `multigrid_batch` — batched (vector forward at the device width, vector reverse) vs per-block streamed, Si + diamond, every level at `25³`, forward and reverse, `to_bits()`; the d-shell fallback; resident vs plain, fused (two-spin) vs single, the fold's continuation; **new:** the forward at widths 1 / 2 / 8 / device vs streamed | **5/5**, every comparison 0e0 (468 s) |
| `multigrid2` (GATE E v2: `get_j`, `nr_rks` LDA vs reference, `int_rho`, thread bit-identity, v1-vs-v2 gap, brute-force periodic products) | **10/10** |
| `multigrid_uks` (v2 two-spin route, M-10) | **5/5** |
| `multigrid_threads` (GATE B v1/v2) | **4/4** |
| `multigrid_level_cache`, `multigrid_memory`, `multigrid_pass2_parallel`, `multigrid_pass2_screen` | **1/1** each |
| `multigrid_cache` (M-02 fingerprint cache, v1 + v2, two cells) | **4/4** |
| `multigrid_scf` (GATE MG-SCF) | **3/3** |
| `multigrid` (GATE E v1) | **6/6** |
| `pyscf-kernels` `multigrid_pair` 8, `pbc_eval_ao_k` 7 | **15/15** |
| `check-dependency-wall` (ALG-06) | PASS |

Not run: GATE A / GATE U (oracle venv), `krks_ksymm`, `ksymm_*`,
`numint_threads`, `eval_ao_*` — nothing in this session touches the AO
evaluation, the numint or the k-symmetry code paths (§5 is a measurement
on the unmodified BEFORE binary); re-run them before the next baseline
capture, as every session of this plan has said of itself.

---

## 5. GATE S — the k-symmetric KRKS, S-03 measured at the gate mesh

Plan-2 §3 S-03 (the IBZ-costed XC quadrature, landed opt-in in session 2 as
`PYSCF_PBC_KSYMM_RHO=symmetrize`, gated by `ksymm_symmetrize_rho` 2/2) has
had "its ratio at the gate mesh not taken" in every session record since.
Taken here on the BEFORE binary (this session changes nothing on this
path), `krks_profile ksymm --driver krks --cell si --mesh 31,31,31 --xc pbe`,
same binary, the arm by environment
(`baselines/2026-09-08-s6-ksymm-krks-si{444,222}-{unfold,symmetrize}-mesh31-pbe.json`,
load 8-9: the multigrid A/B's tail):

| k-mesh | arm | ksymm `kernel()` | warm ksymm `get_veff` | ksymm cold tables | peak RSS | ksymm `e_tot` |
|---|---|---|---|---|---|---|
| 4×4×4 (64 → 20 k) | `unfold` (default) | 5 923 ms | 215 ms | 64k deriv0 905 ms, 64k deriv1 3 050 ms | 3 929 MiB | −7.8749677811124466 |
| 4×4×4 | **`symmetrize`** | **3 932 ms (1.51×)** | **159 ms (1.36×)** | 64k deriv0 911 ms, **20k deriv1 1 327 ms** | **3 681 MiB** | −7.874967781112447 |
| 2×2×2 (8 → 4 k) | `unfold` | 1 902 ms | 37.1 ms | 8k 326 ms, 8k 716 ms | 940 MiB | −7.785668903524661 |
| 2×2×2 | `symmetrize` | 1 807 ms (1.05×) | 32.2 ms | 8k 322 ms, **4k 656 ms** | 850 MiB | −7.785668903524661 |

(The full-BZ `kernel()` beside them, 7 151-7 169 ms at 4×4×4 and 2 529-2 538
ms at 2×2×2, is the same-process control: unchanged by the flag.)

* At 4×4×4 the symmetrised route is **1.51× on the whole k-symmetric SCF**
  and −248 MiB, and the converged energy moves by **≤ 4e-16 Ha** (the two
  arms agree to every printed digit at 2×2×2 and to the 16th at 4×4×4) —
  four orders inside the 1e-11 gate the item was written against. The
  route's GGA table is built at the 20 IBZ points (1.3 s instead of 3.1 s)
  and the per-cycle quadrature runs at 20 points.
* **What it does not shrink:** the deriv-0 table FFTDF builds for `get_j`
  at all 64 k-points (0.9 s, 244 MB) — the J build takes the unfolded
  density, so the full-BZ AO table is still evaluated once. Building J from
  the symmetrised real-space density (which the route already has) would
  remove the last full-BZ table from a pure-functional ksymm SCF; it is a
  ksymm-`get_veff` restructure that changes results at the same 1e-16
  level, listed in §6 as S-10.
* **Default flipped (user decision, 2026-09-08, recorded here as
  D-PBC-33):** `symmetry_rho_enabled` now returns true unless
  `PYSCF_PBC_KSYMM_RHO=unfold`; the old route stays reachable by that
  spelling and `ksymm_symmetrize_rho` compares the two explicitly. Gates
  re-run on the new default: §5.1.

### 5.1 Gates on the flipped default

`crates/pyscf-pbc-dft/src/numint.rs::symmetry_rho_enabled` and
`tests/ksymm_symmetrize_rho.rs` (its unfold arms now spell
`PYSCF_PBC_KSYMM_RHO=unfold`); no-LTO `target/gate`, one scope each:

| gate | result |
|---|---|
| `ksymm_symmetrize_rho` (symmetrize vs unfold, LDA + GGA, RKS + UKS, `nelec` 1e-12 / `e_tot` 1e-11; open shell) | **2/2** |
| `ksymm_threads` (GATE B ksymm, thread bit-identity on the new route) | **2/2** |
| `numint_threads` (GATE B numint) | **1/1** |
| `ksymm_band_ao_reuse` (S-07 / S-08 subset reuse still fires) | **3/3** |
| `ksymm_trace_precision` | **3/3** |
| `kuks_bands` | **2/2** |
| `krks_ksymm` (GATE C: IBZ vs full BZ, KRKS / KUKS / Hubbard, GDF band route) | **7/7** + 3 ignored |

`krks_profile ksymm` still takes the route from the environment, so the
`unfold` rows of §5 are reproducible with `PYSCF_PBC_KSYMM_RHO=unfold`.

---

## 6. Not done, and why

* **The `exp` is now half of both pair kernels** (§3, kill-switch arm):
  1.73 G scalar `cube_math` calls per direction per level-3 sweep. Two
  levers, neither taken: (a) a `Vector<f64, N>` `exp` in `cube-math`
  (the same glibc schedule per element with a per-element table gather —
  bit-exact by construction, a cube-math item, not a pyscf one); (b) the
  per-point instance-radius screen (`r² > r_inst²` → skip), which drops
  terms below `cell.precision` and therefore changes results — an opt-in
  item under RULE S, expected ≥ 2× on both directions from the block-vs-
  ball volume ratio, to be measured and landed alone.
* **Uniform-set fast path in the vector reverse** (85 % of occurrences sit
  in same-set runs ≥ 8): the predication could give way to the scalar
  `while` power loops when a group's N lanes share a set, saving ~6
  vector ops per slot per point. Modelled ≤ 1.3× on the reverse's non-`exp`
  half; not measured.
* **GPU width policy**: the forward kernel's width is the device's f64
  vector width; on a discrete GPU that is fewer lanes, not more — pin
  `PYSCF_MG_PAIR_LINE=1` there until measured (UNVERIFIED, no f64 GPU here).
* **S-03 default flip / S-10** (§5) — a planning decision.
* **The two-spin `out_rho_b` / `out_integrate_b` buffers** are still
  per chunk (a pre-existing shape, not shared through the level scratch):
  +48 MB per level-3 chunk on a KUKS v2 run. Not measured.
* **GATE A / GATE U** — not run (need `PYSCF_ORACLE_VENV=1`); the v2 SCF
  energy is bit-identical to the pre-session binary's, so the oracle rows
  are unaffected by construction.

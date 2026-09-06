# KRKS + k-symmetry + multigrid — session 3 execution record

**Plan:** [`KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md`](./KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md)
(session 2's record: [`KUKS-KSYMM-MULTIGRID-SESSION-2-EXECUTION-SUMMARY.md`](./KUKS-KSYMM-MULTIGRID-SESSION-2-EXECUTION-SUMMARY.md)).
**Date:** 2026-09-06.
**Machine:** 16 cores, 30 GiB, CubeCL **CPU** runtime (`pyscf-algebra`
`default = ["cpu"]`; the ROCm iGPU has no f64). Every number is a CPU-runtime
number; every GPU claim stays UNVERIFIED (plan-2 backend note, RULE G/T).
**Load:** another session's Phase-16 measurement (`m2_spread.py`, ~4 cores)
ran for most of this session; the 1-minute load average is printed inside
every report and quoted beside every number. RULE O therefore allows only
**within-run shares and same-binary A/B ratios** here — no row below goes
into `baselines/` (P-10's guard refuses above load 4.0, by design).

---

## 0. What session 2 left, and what this session did

Session 2 landed every plan-2 item's *code* and none of its *measurements*.
This session ran the instruments first (plan-2 §4: "instruments before
items"), read what they said, and then landed the three items the numbers
pointed at — none of which was in plan 2, because plan 2 was written before
the instruments existed.

| step | result |
|---|---|
| A-00 stage table (Q9) | **MEASURED**, §1 |
| M-01 instrument, v1 and v2, mesh 11³ and 25³ | **MEASURED**, §2 |
| ksymm `kernel()` / `get_veff` ratios at the gate mesh | **MEASURED**, §3 |
| **S-07** (new): serve the IBZ band AO table from the full-BZ table | landed, bit-exact, §3 |
| **M-11** (new): keep v1's collocated level values for the SCF's life | landed, bit-exact, §2 |
| **M-12** (new): v2 per-slot arrays replaced by one index — RULE T | landed, bit-exact, §2 |
| **S-08** (new): the same subset reuse in the FFTDF AO cache (`Arc`-shared) | landed, bit-exact, §5.3-5.4 |
| **K-08b** (new): scatter accumulate with the `k` loop inside the lane; dense path for fully-kept images | landed, bit-exact, §5.5-5.6: cold AO pass **1.3-1.65× faster** |
| **S-09** (new): `kpts_ibz` as the bitwise copy of its full-BZ points — what lets S-07/S-08 fire | landed, re-gated (not bit-exact by construction), §5.4/5.7: ksymm SCF **1.26× → 0.90×** of full BZ |
| GATE MG-SCF | first green run; its KUKS fixture assertion rewritten, §4.2 |

---

## 1. GATE AO — the A-00 stage table (Q9 answered)

`krks_profile ao --cell si --nk 2,2,2 --mesh 31,31,31 --deriv {0,1}`, W-09
screen on, cold `eval_ao_kpts`, load 7.4-9.8:

| cell | deriv | `n` (reals) | cold ms | shift+pack | **eval_gto** | scatter | **K-08** | peak RSS |
|---|---|---|---|---|---|---|---|---|
| si gth-szv 2×2×2 | 0 | 238 328 | 2 683 | 1.7 % | **68.8 %** | 0 (A-02) | 23.1 % | 256 MiB |
| si gth-szv 2×2×2 | 1 | 953 312 | 9 994 | 0.6 % | **74.2 %** | 0 | 23.1 % | 516 MiB |
| si gth-dzvp 2×2×2 | 1 | 3 098 264 | 58 412 | 0.1 % | **75.2 %** | 0 | 24.0 % | 1 301 MiB |

* `nimgs = 1331` (11³) on every row — the image LIST; the instrument now also
  prints how many images the screen actually lets through and how many grid
  points those launches cover (`launched_images`, `kept_points_total`,
  `kept_fraction`), which the plan's Q9 could not see. Filled in §5.
* A-02's claim holds: the scatter stage is gone and the RULE-T round trip is
  0 bytes per image (`legacy round-trip` 3.8 / 15.3 / 49.6 MB per image).
* The kernel stage is ~3× the Bloch accumulate on every row, so A-03
  (vectorise the AO kernel's grid axis) is the next AO item by the plan's own
  DEFER UNTIL clause. Not started this session; see §6.

---

## 2. GATE MG — the M-01 instrument

`krks_profile multigrid --driver krks --numint {grid,v1,v2}`, Si gth-szv,
LDA, converged `kernel()` then ONE warm `get_veff`, load 19-21 (the other
session's job restarted during this block; ratios within a run only):

| mesh | numint | `kernel()` ms | warm `get_veff` ms | peak RSS MiB | `e_tot` |
|---|---|---|---|---|---|
| 11³ | grid (reference) | 1 529 | 1.6 | 180 | −7.158713437125394 |
| 11³ | v1 | 1 611 | 23.3 | 185 | −7.158713437125398 |
| 11³ | v2 | 19 769 | 2 866 | **2 248** | −7.15871321455335 |
| 25³ | v1 | 2 493 | 191 | 210 | −7.160554197438174 |
| 25³ | v2 | 42 184 | 11 621 | **4 390** | −7.160554062714283 |

### 2.1 v1 — where the 191 ms went (M-11's WHY)

The per-level spans (forward 4.7 ms, reverse 3.6 ms, two FFTs 1.0 ms) sum to
under 10 ms of the 191 ms warm `get_veff` at 25³. The remaining ~180 ms sat
outside every span: `collocate_all_levels`, which re-ran the v1 collocation
kernel on every SCF cycle although its input is the cell alone (the density
matrix enters only through `level_rho` / `level_pass2`). M-02 had shared one
collocation between the two directions of ONE call; nothing kept it across
calls. Session 3 added the `pbc_mg_collocate` / `pbc_mg_xc_parts` spans to
the instrument and then **M-11** (§2.3).

### 2.2 v2 — where the 4.4 GiB went (M-12's WHY)

Per-level rows at 25³ (warm `get_veff`, forward / reverse):

| level | launches | forward ms | reverse ms | RULE-T bytes/call (post-M-06) |
|---|---|---|---|---|
| 1 | 1 / 1 | 10.5 | 13.3 | 4.1 MB |
| 2 | 4 / 4 | 1 311 | 1 417 | 602 MB |
| 3 | 13 / 13 | 4 267 | 4 599 | **1 866 MB** |

1.87 GB of per-call transfer at level 3 is `slot_coef` — one `f64` per
CONCATENATED slot, gathered on the host (`kcoef[slot_global[s]]`) and
uploaded per chunk per direction, on top of a resident `slot_pow` (`u32` per
slot) and the host copies of both. The reverse direction uploaded a
`vec![1.0; nslots]` — a constant. Per-slot data is what the 256 MiB chunk
budget is spent on, so per-slot bytes ARE v2's memory. See §2.4.

### 2.3 M-11 — v1 level values cached with the task list (**bit-exact**)

`crates/pyscf-pbc-dft/src/multigrid/numint.rs`: `V1Tasks` (the M-02
cell-fingerprinted cache) gains `level_values: OnceLock<Vec<LevelValues>>`,
filled on first use and served on every later cycle; `nr_rks`, `nr_uks`,
`get_j`, `eval_rho_g` all read it. The memory rule is unchanged — a cell over
`0.25 · max_memory` streams level by level and caches nothing, exactly as
before. Kill switch `PYSCF_PBC_MULTIGRID_LEVEL_CACHE=0`.

**Bit-parity: EXACT** (the same table read instead of rebuilt). Asserted by
`tests/multigrid_level_cache.rs`: miss vs hit vs uncached, `nr_rks` /
`nr_uks` / `get_j`, two cells, `to_bits()`.

### 2.4 M-12 — v2 per-slot arrays → one `u32` index (**bit-exact; RULE T**)

`crates/pyscf-kernels/src/multigrid_pair.rs`,
`crates/pyscf-pbc-dft/src/multigrid/pair.rs`. Per concatenated slot the
batch carried `slot_pow: u32` (resident) and `slot_coef: f64` (per call);
both are functions of the level's KERNEL slot `k = slot_global[slot]`:
`slot_pow[slot] == pack(kslot_pow[k])` and `slot_coef[slot] == kcoef[k]`
(forward) or `1.0` (reverse). The batched kernels now read
`slot_global[slot]` (resident `u32`) and index two SMALL per-kernel-slot
tables — `kpow` (resident, `nkslots · 4` B) and `kcoef` (per call,
`nkslots · 8` B); the reverse kernels drop the coefficient entirely
(`1.0 · x == x` exactly, so removing the multiply is bit-exact).

RULE-T ledger, per chunk per direction: **before** `nslots · 8` (coef upload)
+ host gather of the same; **after** `nkslots · 8`. Resident per slot: 4 B
(`slot_global` on the device) + 4 B (its host copy, needed by the reverse
scatter-add) — was 4 (device `slot_pow`) + 4 (host `slot_pow`) + 4 (host
`slot_global`) + 16 transient (host + device coef). Numbers in §5.

**Bit-parity: EXACT** — the same values reach the same operations in the
same order; `multigrid_batch.rs` (batched vs streamed, 0e0) is the assertion.

---

## 3. GATE S — ksymm at the gate mesh, and S-07

`krks_profile ksymm --driver krks --cell si --nk 2,2,2 --mesh 31,31,31 --xc pbe`,
load 13.8:

| | full BZ | ksymm | ksymm / full |
|---|---|---|---|
| `kernel()` | 14 535 ms | 17 671 ms | **1.216** |
| warm `get_veff` | 56.6 ms | 39.0 ms | 0.689 |
| `e_tot` | −7.785668903669 | −7.785668903525 | (RULE K: not a comparison) |

The per-cycle ratio matches session 1's 0.73×; the WHOLE SCF is slower under
symmetry. Cause (VERIFIED by reading, then MEASURED with the new
`pbc_eval_ao_kpts` counter, §5): `KsymAdaptedKrks::get_veff_tagged` hands the
quadrature `kpts_band = kpts_ibz`, and `KNumInt::nr_rks` / `nr_uks`
(`numint.rs:843-847`) evaluate a SECOND AO table at the band points —
`eval_ao_kpts` again, whose `eval_gto` stage (§1: 69-75 %) does not shrink
with the k-count — and cache it as a second `comp·ngrids·nao·16·N_ibz`-byte
entry. The IBZ points are a subset of the full-BZ list the density was just
evaluated over (`ibz2bz`).

### S-07 — serve the band table from the sampling table (**bit-exact**)

`crates/pyscf-pbc-dft/src/numint.rs`: `band_subset_map` finds every band
point (bitwise) in `self.kpts`; when all are found, `ao1 = ao2` and the vxc
accumulator reads `ao2.at(map[k])`. K-08 accumulates every k independently
(`out[k·n+p] += phase_k · ao[p]`, same AO block, same phase whichever list it
is launched with), so the entry is bit-identical. A band point off the list
takes the old path. Kill switch `PYSCF_PBC_BAND_AO_REUSE=0`.

**Bit-parity: EXACT**, asserted by `tests/ksymm_band_ao_reuse.rs` (reuse on
vs off, LDA and GGA, `nr_rks` and `nr_uks`, every `vmat`/`nelec`/`excsum`
at `to_bits()`). Memory: one AO table instead of two (61 MB at deriv 1,
mesh 31, szv; 200 MB dzvp). Numbers in §5.

---

## 4. Gates re-run

Built in a separate no-LTO target directory (`CARGO_TARGET_DIR=target/gate
CARGO_PROFILE_RELEASE_LTO=false`), because every thin-LTO release test
binary here links for ~10-30 min against the ~500 `libxc_rkernel` crates and
`cargo build -p pyscf-bench` / `cargo test -p pyscf-pbc-dft` unify features
differently, each rebuilding that tree (memory
`libxc-release-builds-are-slow`). Bit-exactness does not depend on LTO —
both arms of every gate are the same binary.

| gate | result |
|---|---|
| `multigrid_batch` (M-12: batched vs streamed, resident vs plain, fused vs single) | **4/4**, 47 `to_bits()` comparisons at 0e0 (Si + diamond, levels 1-3 at 25³, incl. the d-shell fallback) |
| `multigrid_level_cache` (M-11) | **1/1** — miss = hit = uncached, `nr_rks`/`nr_uks`/`get_j`, two cells |
| `ksymm_band_ao_reuse` (S-07 rks/uks, S-08 FFTDF vj/vk, off-list band) | **3/3** (171 s) |
| `eval_ao_stages` (GATE AO: thread bit-identity, screen ≤ 1e-11, **K-08b** byte-equal) | **2/2** |
| `multigrid_threads` (GATE B v1/v2) | **4/4** (139 s, on the M-11/M-12 binaries) |
| `pyscf-kernels`: `multigrid_pair` 8, `pbc_eval_ao_k` 7, `eval_gto_oracle` 6, `eval_gto_lge1` 4 | **25/25** |
| the remaining plan-2 §5 rows | §4.1 |

### 4.1 Remaining rows

| gate | result |
|---|---|
| `multigrid_uks` (M-10 fused vs single, v1/v2 vs reference, closed-shell identity) | **5/5** |
| `multigrid_pass2_parallel` | **1/1** |
| §5.7 lints: `check-dependency-wall` (ALG-06), `check-orphan-modules` (367 files), `check-no-fma` (7 asm files) | **PASS / PASS / PASS** |
| `multigrid2` | see the memory note below; per-test runs in §4.2 |
| `multigrid_cache`, `multigrid_scf`, `multigrid`, `multigrid_memory`, `krks_ksymm`, `ksymm_threads`, `numint_threads`, `ksymm_symmetrize_rho`, `multigrid_threads`, `multigrid_batch` | §4.2 |

**Two things the gate runs taught, both about the harness, not the code:**

* A `ulimit -v` around a gate binary makes EVERY CubeCL CPU-runtime launch
  panic (`cubecl-cpu worker.rs:36` / `queue.rs:38` / `client.rs:105
  CallError`) — 30 false failures in one pass. The runtime spawns 64 MB-stack
  worker threads plus the MLIR JIT, so an address-space cap starves it long
  before real memory is used. Guard with `systemd-run --user --scope -p
  MemoryMax=…` instead (memory `ulimit-v-breaks-cubecl-cpu-runtime`).
* `multigrid2` cannot run as ONE process on this box any more: every test
  builds its own `MultiGridNumInt2` tables (two cells at 25³, ~4 GB of
  resident batches each) and the CubeCL memory pool retains freed buffers,
  so RSS climbs test by test — 13 GB after four tests, killed at a 24 GB cap
  in the sixth. Session 2 ran it on an otherwise empty 30 GB box. §4.2 runs
  it one test function per process (fresh pool each), same assertions.

### 4.2 Per-gate runs under `systemd-run` scopes

| gate | result | peak RSS |
|---|---|---|
| `multigrid_uks` | **5/5** | — |
| `multigrid_cache` (M-02 fingerprint cache, v1 + v2, two cells) | **4/4** (108 s) | — |
| `multigrid` (GATE E v1: `get_j`, `nr_rks` LDA, `int_rho`, thread identity, speed ratio) | **6/6** | — |
| `multigrid_memory` (forced-low `PYSCF_MAX_MEMORY` streaming vs retained, bit-exact) | **1/1** | — |
| `multigrid_scf` (GATE MG-SCF) | KRKS row **pass**; KUKS row: see finding below | 2.1 GB |
| `krks_ksymm` (GATE C: IBZ vs full BZ, KRKS/KUKS/Hubbard, GDF band route) | **7/7** + 3 ignored (263 s) | — |
| `ksymm_threads` (GATE B ksymm) | **2/2** | — |
| `numint_threads` (GATE B numint) | **1/1** | — |
| `ksymm_symmetrize_rho` (S-03 route vs unfold, LDA+GGA, open shell) | **2/2** | — |
| `multigrid_threads` (GATE B v1/v2, final binaries) | **4/4** (117 s) | — |
| `multigrid_batch` (M-12 identity, final binaries) | **4/4** (152 s) | — |
| `multigrid2` (GATE E v2), one test function per process | **10/10** | 0.07-6.9 GB per test; `v2_rho_g_matches_v1_with_and_without_the_ladder` **14.3 GB** |

**GATE MG-SCF finding — the KUKS row had never run green.** The multigrid
arms converge to the grid arm's energy (v1: `|E_mg − E_grid| = 3.6e-15`), but
the test's RULE U assertion `max |dm_a − dm_b| > 1.0` — the P-01 idiom copied
from the Si/NiO fixtures — cannot be met by the lithium/sto-3g fixture, whose
converged spin-density difference is 0.205 in EVERY arm, grid included. No
planning record shows this test green (session 2 listed GATE MG-SCF as "not
yet reported"). The assertion is replaced by two that mean something: the
grid arm's own `max |dm_a − dm_b| > 0.1` (genuinely open-shell), and each
multigrid arm's value equal to the grid arm's within `10 · tol` (the same
spin state). Re-run in §4.3.

### 4.3 After the S-09 rebuild (every ksymm-affected gate, one scope each)

| gate | result |
|---|---|
| `ksymm_band_ao_reuse` — now with the bitwise-subset precondition asserted, so the reuse genuinely fires on both arms' comparison | **3/3** |
| `pyscf-pbc-symm`: `kpts_ibz` 5, `kpts_transform` 16, `ktensor` 19, `ktensor_ksymm_scf` 1 | **41/41** |
| `pyscf-pbc-scf`: `khf_ksymm` (S-02 band route, GDF/FFTDF) | **6/6** |
| `krks_ksymm` (GATE C) | **7/7** + 3 ignored |
| `ksymm_threads`, `ksymm_symmetrize_rho`, `numint_threads` | **2/2**, **2/2**, **1/1** |
| `multigrid_scf` (GATE MG-SCF, with the §4.2 assertion) | **3/3** — KRKS v1/v2 at their floors; KUKS grid `max|dm_a−dm_b|` 0.205, v1 `|ΔE|` 3.6e-15, v2 1.5e-7, both the grid arm's spin state |
| `eval_ao_stages` (GATE AO incl. K-08b) on the S-09 build | **2/2** |

Not run this session: GATE A / GATE U (`tests/gate.rs`, `gate_openshell.rs`)
need `PYSCF_ORACLE_VENV=1` and the upstream interpreter. Every item here is
either bit-exact (S-07, S-08, M-11, M-12, K-08b — each with its own
`to_bits()` gate above) or confined to the k-symmetric drivers (S-09, gated
by GATE C 7/7, `khf_ksymm` 6/6 and the 41 pbc-symm rows), so the full-BZ
oracle rows are unaffected by construction; they should still be re-run
before the next baseline capture.

## 5. AFTER measurements

Same binary as the BEFORE arm where a kill switch exists (one variable per
ratio); the no-LTO build for everything in this section, so absolute times are
NOT comparable to §1-§3's LTO numbers — the within-run ratios are.

### 5.1 GATE AO — the screen's yield (Q9, second half)

`krks_profile ao --cell si --nk 2,2,2 --mesh 31,31,31`, load 0.3-3.5:

| deriv | screen | launched images | kept points | kept / (nimgs·ngrids) | cold ms | eval_gto | K-08 |
|---|---|---|---|---|---|---|---|
| 0 | on | **454** / 1331 | 9.43 M | 0.238 | 2 558 | 1 747 (68.3 %) | 584 (22.8 %) |
| 1 | on | 454 / 1331 | 9.43 M | 0.238 | 9 071 | 6 660 (73.4 %) | 2 140 (23.6 %) |
| 1 | off | 1331 | 39.65 M | 1.000 | 13 651 | 11 054 (81.0 %) | 2 100 (15.4 %) |

Read off the table:

* The screen drops 66 % of the images but keeps 70 % of the grid on the ones
  it launches (20.8 k points per launch). Per launched image the kernel stage
  is 3.85 ms (deriv 0) — **46 ns per (point, shell) lane** on 16 threads, ~15
  primitive `exp`s each: the per-lane cost is the software `exp`, not
  dispatch (A-01 already removed that). A-03 (vectorising the grid axis) is
  the AO item that remains, and it is a kernel re-shape, not a hoist.
* **K-08 costs the same 2.1 s with the screen on and off** although it
  touches 4.2× fewer elements — the scatter variant (`accumulate_device_scatter`,
  one lane per `(k, element)` with five integer divisions and two
  read-modify-writes) is ~4× less efficient per element than the dense
  vectorised one. That is a bit-exact item for the next session: one lane per
  element with the `k` loop inside (the accumulation per `(k, p)` is still
  one addition per image, so the order is unchanged), and the dense path for
  images whose every block is kept.

### 5.2 GATE MG — M-11 and M-12, same binary, kill-switch A/B

`krks_profile multigrid --driver krks --numint {v1,v2} --mesh 25,25,25`,
Si gth-szv LDA, load 5.2-5.5:

| item | arm | `kernel()` | warm `get_veff` | collocations in `kernel()` | peak RSS | `e_tot` |
|---|---|---|---|---|---|---|
| **M-11** | cache OFF (`PYSCF_PBC_MULTIGRID_LEVEL_CACHE=0`) | 2 257 ms | **172.4 ms** (collocate 161.2 ms of it) | 3 × 175 ms | 209 MiB | −7.160554197438174 |
| **M-11** | cache ON | 1 915 ms | **12.5 ms** | 1 × 185 ms | 195 MiB | −7.160554197438174 |

Warm v1 `get_veff` **13.8× faster**, bit-identical energy, and the collocation
kernel now runs once per SCF. v1's per-cycle cost is now the two level
sweeps (4.6 + 3.9 ms) and the XC middle (3.8 ms).

| item | arm | warm `get_veff` | level-3 RULE-T bytes / call | level 3 slots / instances / kslots | peak RSS |
|---|---|---|---|---|---|
| **M-12** | BEFORE (§2, LTO build, load 19) | 11 621 ms | 1 866 MB | — | 4 390 MiB |
| **M-12** | AFTER (no-LTO, load 5.2) | 9 842 ms | **766 MB** | 77.8 M / 13.8 M / 1.38 M | **3 962 MiB** |

The remaining 766 MB per call at level 3 is the reverse read-back
(`nslots · 8` = 622 MB) plus the two weight uploads; the per-slot
coefficient (622 MB up, and its host gather) is gone, as is the reverse
`vec![1.0; nslots]`. The level's kernel-slot count is 1.38 M against 77.8 M
concatenated slots (56×), which is why indexing through `slot_global` was the
right cut. What is left of v2's memory is per-instance data (13.8 M instances
× 40 B, host + device) and the per-slot lists (`slot_global` host + device,
`block_sel`); the next memory item is dropping the host copies once the
device copy exists.

### 5.3 GATE S — S-07 and the second site

`krks_profile ksymm --driver krks --cell si --nk 2,2,2 --mesh 31,31,31 --xc pbe`,
same binary, load 9-12 (the other session's job returned):

| arm | full `kernel()` | ksymm `kernel()` | ksymm/full | cold `eval_ao_kpts` calls inside ksymm `kernel()` |
|---|---|---|---|---|
| reuse OFF | 12 853 ms | 18 235 ms | 1.419 | 4 calls, 17 285 ms, k-counts `[4, 8, 4, 8]` |
| reuse ON (S-07) | 12 775 ms | 17 452 ms | 1.366 | 4 calls, 16 537 ms, k-counts `[4, 8, 4, 8]` |

S-07 is bit-exact and gated, but the counter shows it is NOT where the
ksymm driver's extra AO tables come from: the two 4-point tables are still
built with the quadrature reuse on. The remaining site is the FFT
density-fitting layer — `Fftdf::ao_kpts(band)` (`fftdf.rs:179`) evaluates its
own table at `kpts_band` for `get_j_kpts` / `get_k_kpts` (`fft_jk.rs:125`,
`:358`), with its own cache. **S-08** (landed after this table): the same
subset map in `Fftdf::ao_kpts`, sharing the sampling table's blocks by `Arc`
(no copy). Its gate is the `fftdf_band_subset_reuse_is_bit_exact_for_j_and_k`
row of `ksymm_band_ao_reuse.rs`; its measurement is §5.4.

### 5.4 S-08 measured — and why neither S-07 nor S-08 fired: S-09

Same binary, idle box (load 1.3), `krks_profile ksymm --driver krks --cell si
--nk 2,2,2 --mesh 31,31,31 --xc pbe`, the counter now printing each cold
table's k-count, eval name and wall:

| arm | ksymm `kernel()` | cold AO tables inside the ksymm `kernel()` |
|---|---|---|
| reuse OFF | 10 910 ms | `4k/GTOval_sph 908`, `8k/GTOval_sph_deriv1 4881`, `4k/GTOval_sph_deriv1 2670`, `8k/GTOval_sph 1507` ms |
| reuse ON (S-07 + S-08) | 10 782 ms | `4k/GTOval_sph 862`, `8k/GTOval_sph_deriv1 4834`, `4k/GTOval_sph_deriv1 2657`, `8k/GTOval_sph 1495` ms |

The two 4-point tables — the FFTDF `GTOval_sph` band table (first call, the
J build) and the quadrature's `GTOval_sph_deriv1` band table (third) — are
still built with the reuse on: **both subset maps refused on every call.**
Cause (VERIFIED, `kpts.rs:243-246`): `make_kpts_ibz` derives
`kpts_ibz = cell.get_abs_kpts(kpts_scaled_ibz)` — faithful to upstream
(`kpts.py:74`) — an abs→scaled→abs round trip that is not a bitwise
identity, so the IBZ list is never a BITWISE subset of the sampling list.
And because the AO block at a k-vector depends on that vector's exact bits
(K-07's `exp(i k·L)` phases), reusing a block at an ulp-different k would
NOT have been bit-exact — the maps were right to refuse. The two gates
passed because both arms took the second-evaluation path and agreed
trivially; the gate now asserts the bitwise-subset precondition first.

**S-09** — `kpts_ibz[i] = kpts[ibz2bz[i]]`, the copy it is defined to be.
Moves each IBZ k-vector by at most a few ulps against upstream's
derivation; the star / little-group bookkeeping uses the SCALED points and
is untouched; every ksymm gate is 1e-11 or looser (GATE C) or a same-binary
bit-identity. `PYSCF_PBC_KPTS_IBZ_ROUNDTRIP=1` restores upstream's
derivation. Not bit-exact against the previous ksymm numbers (an ulp-level
k-vector change is an arithmetic change — RULE S says so plainly), which is
why it carries the switch and is the one item here re-gated rather than
asserted identical. Measurement: §5.7.

### 5.7 S-09 + S-07 + S-08 measured — the ksymm SCF is now faster than the full one

`krks_profile ksymm --cell si --mesh 31,31,31 --xc pbe`, same binary, load
10.8-11.6 (the other session's job was back; the A/B is same-binary and the
cold-table COUNT is load-independent):

| driver | k-mesh | arm | full `kernel()` | ksymm `kernel()` | **ksymm / full** | cold AO tables in the ksymm `kernel()` | warm `get_veff` ksymm/full | peak RSS |
|---|---|---|---|---|---|---|---|---|
| KRKS | 2×2×2 | reuse OFF | 8 626 ms | 10 865 ms | 1.260 | **4** (`4k` 908 + `8k` 4 823 + `4k` 2 665 + `8k` 1 519 ms) | 0.683 | 659 MiB |
| KRKS | 2×2×2 | **reuse ON** | 8 424 ms | **7 557 ms** | **0.897** | **2** (`8k` 1 582 + `8k` 5 009 ms) | 0.769 | 615 MiB |
| KUKS | 2×2×2 | reuse ON | 8 845 ms | **7 535 ms** | **0.852** | 2 | 0.737 | 614 MiB |
| KRKS | 4×4×4 | reuse ON | 45 750 ms | **43 112 ms** | 0.942 | 2 (`64k` 9 340 + `64k` 31 657 ms) | **0.501** | 3 456 MiB |

* The two 4-point tables are gone (3.6 s of a 10.9 s KRKS SCF); the ksymm
  `kernel()` ratio goes from **1.26 → 0.90** on the same binary, and the
  k-symmetric run is faster than the full-BZ one for the first time in this
  plan's history (session 1 measured 1.2, §3 measured 1.22-1.37). Energies
  are unchanged to the printed 12 digits on every row (RULE K: the two
  columns are not a correctness comparison, the ksymm `e_tot` is the same
  before and after S-09 to 1e-12).
* What is left at 2×2×2: the ksymm SCF still evaluates its two tables at the
  FULL 8 k-points (the Group-A unfold, plan-2 §2.2) — S-03's opt-in
  `symmetrize` route is the item that would shrink them to `N_ibz`.
* At 4×4×4 the whole run is 92-93 % cold AO evaluation (42 of 45.8 s) on
  both drivers; the ksymm advantage there is the per-cycle 0.50× and the
  saved band tables (2.7 s). The AO kernel (§5.1: 46 ns per lane, software
  `exp`) is the remaining lever — A-03.
* Warm `get_veff` ksymm/full moved 0.68 → 0.77 at 2×2×2 with reuse on: the
  warm number of the reuse-OFF arm was flattered by the SECOND table's cache
  hits landing in the timed call; the absolute ksymm `get_veff` is 37-38 ms
  either way.

### 5.5 K-08b — the Bloch accumulate on the screened path (**bit-exact**)

`crates/pyscf-kernels/src/pbc/eval_ao_k.rs`,
`crates/pyscf-pbc-gto/src/eval_gto.rs`. Two changes, both from §5.1's
finding that K-08 cost 2.1 s whether the screen kept 24 % or 100 % of the
grid:

1. `eval_ao_k_accumulate_scatter_kloop_kernel` — one lane per kept AO
   element with the `nkpts` accumulations inside it, instead of one lane per
   `(k, element)`: the five-division index decode and the `ao[q]` /
   `index[j]` loads happen once per element, not `nkpts` times. Each `(k, p)`
   accumulator still receives exactly one addition per image
   (`pr[k] · ao[q]`, the same product into the same place), so the value is
   bit-identical. `PYSCF_PBC_K08_SCATTER=legacy` pins the old kernel.
2. An image whose EVERY block the screen keeps is handed to the dense path
   (contiguous shift, the vectorised K-08) instead of being gathered point by
   point and scatter-accumulated back. The gathered "sub-grid" was the whole
   grid in grid order, so the dense kernel performs the same additions.
   `PYSCF_PBC_AO_DENSE_FULL=0` keeps the gather/scatter path.

**Bit-parity: EXACT**, asserted by
`eval_ao_stages.rs::k08b_scatter_and_dense_full_images_are_bit_exact` (the
whole screened AO table at 8 threads, new path vs both switches flipped,
byte-equal). Measurement: §5.6.

### 5.6 K-08b measured — same binary, idle box (load 1.3-1.9)

`krks_profile ao --cell si --nk 2,2,2 --mesh 31,31,31`, screen on; the A arm
is `PYSCF_PBC_K08_SCATTER=legacy PYSCF_PBC_AO_DENSE_FULL=0`:

| cell | deriv | launched / kept | cold `eval_ao_kpts`, legacy | cold, **K-08b** | ratio | K-08 stage, legacy → K-08b |
|---|---|---|---|---|---|---|
| si gth-szv | 0 | 454 / 0.238 | 2 604 ms | **1 994 ms** | **1.31×** | 604 → 398 ms |
| si gth-szv | 1 | 454 / 0.238 | 8 873 ms | **5 384 ms** | **1.65×** | 2 085 → 1 080 ms |
| si gth-dzvp | 1 | 489 / 0.249 | 29 407 ms | **18 198 ms** | **1.62×** | 6 693 → 3 574 ms |

The whole-pass gain (1.3-1.65×) is LARGER than the K-08 stage's own share
(20-23 %) because the stage spans are host-side and the runtime is lazy
(`05_lazy_execution.md`): a launch returns before the kernel runs, so part
of the legacy scatter kernel's cost was being clocked inside the NEXT
image's `eval_gto` span. §1's stage table therefore under-attributed K-08;
the same-binary A/B on the cold total is the number to quote. Peak RSS is
unchanged (515 MiB at deriv 1). On the GPU runtime the same change removes
`nkpts − 1` index decodes and loads per element; the gain there is
UNVERIFIED (RULE G).

## 6. Not done, and why

* **A-03** — after K-08b the AO kernel stage IS the cold pass (§5.1: 46 ns
  per `(point, shell)` lane, ~15 software `exp`s each; at 4×4×4 the cold
  tables are 92 % of the SCF, §5.7). It re-shapes a 2 500-line `f64` kernel;
  the launched-image and kept-point numbers it needed are now in §5.1.
* **S-03 default** — the ksymm SCF still evaluates its two tables at the full
  `N` points (Group-A unfold); the opt-in `symmetrize` route is gated
  (`ksymm_symmetrize_rho` 2/2) but its ratio at the gate mesh was not taken.
* **P-10 baselines** — still owed; the 1-minute load never sat under 4.0
  for a full ksymm run (the idle window at 11:45 went to the AO / multigrid
  A/B, §5.1-5.2).
* **v2 memory, next cut** — M-12 removed the per-slot coefficient; what
  remains is per-instance data (13.8 M × 40 B, host + device) and the host
  copies of `slot_global` / `block_sel` beside their device twins. Dropping
  the host geometry once the device copy exists halves v2's resident set on
  the CPU runtime. v2 speed is untouched (RULE M).
* **GATE A / GATE U** — not run (§4.3).
* **`cargo test` shape** — every gate here ran from the no-LTO
  `target/gate` binaries under `systemd-run` scopes; the plan's §5 command
  lines (thin-LTO `--release`) still cost ~30 min per test binary on this
  box and cannot run `multigrid2` as one process (§4.1).

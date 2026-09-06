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

## 4. Gates re-run (fill in)

## 5. AFTER measurements (fill in)

## 6. Not done, and why

* **A-03** — the AO kernel stage is 69-75 % of the cold pass on every row
  and is the next lever; it re-shapes a 2 500-line `f64` kernel and needs the
  launched-image / kept-point numbers (§5) to size the gain first.
* **P-10 baselines** — still owed; the machine was never idle (load 7-21).
* **v2 kernel speed** — M-12 removes bytes, not flops; v2 stays 60× v1 at
  25³ (RULE M: it exists for the memory shape and `isinstance`).

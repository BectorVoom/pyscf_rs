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

# 17-06 MEASUREMENT — the `KsymmArray` incore / out-of-core crossover

**Measured:** 2026-09-01, `cargo test -p pyscf-pbc-symm --release --test ktensor
-- --nocapture measure_the_incore_outcore_crossover`.
Machine: the phase's usual host, CPU backend (`PYSCF_BACKEND` unset).
**Run the measurement ALONE.** With the rest of the file's tests running
concurrently the same table moves by up to ~2x in either direction (one
contended run even reported a 0.16x "outcore is faster" row); the numbers
below are from an isolated `-- measure_the_incore_outcore_crossover` run.

## What was asked, and what the question actually is

17-06-PLAN.md Task 1: *"measure, do not guess, where the incore/out-of-core
crossover should sit … at a few tensor sizes spanning the boundary upstream
picks (`ktensor.py:54-83`'s own heuristic)"*.

**`ktensor.py:54-83` has no heuristic.** `incore` is read straight off the
metadata dict — `incore = metadata.get('incore', True)` (`:50`) — and every
decision is made by the CALLER, always as a pure MEMORY test:

```
kccsd_rhf_ksymm.py:153-155   mem_now = lib.current_memory()[0]
                             if (cc.incore_complete or
                                 _memory_4d(cc, [nocc,]*4) + mem_now < cc.max_memory * .9):
```
and the same shape at `:164-166`, `:218-220`, `:233-235`, `:289-290`,
`:304-310`, `:321-323`, `:413-417`. There is no size constant anywhere to
copy into Rust.

So the honest question is not *"where is upstream's threshold"* but
**"what does forcing the out-of-core path cost when the tensor would have
fit"** — which is exactly what decides whether the Rust port needs a
minimum-size floor bolted on top of the memory test.

## The measurement

`si`, `[2,2,2]`, full space group. `nkpts = 8`, `nkpts_ibz = 3`,
`len(kqrts_ibz) = 50` stored rank-4 blocks; the unfold materialises all
`nkpts^3 = 512` blocks. Subarray `[nocc, nocc, nvir, nvir]`, `label = 'oovv'`,
`trans = 'nncc'`. Both paths asserted BIT-identical at every size.

| `nocc`/`nvir` | stored (MiB) | `from_raw` incore | `from_raw` outcore | ratio | unfold incore | unfold outcore | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1/1 | 0.0008 | 0.0000 s | 0.0010 s | **655x** | 0.0001 s | 0.0001 s | **1.24x** |
| 2/2 | 0.0122 | 0.0000 s | 0.0005 s | **217x** | 0.0001 s | 0.0002 s | **1.62x** |
| 3/4 | 0.1099 | 0.0000 s | 0.0009 s | **224x** | 0.0003 s | 0.0004 s | **1.24x** |
| 4/6 | 0.4395 | 0.0001 s | 0.0009 s | **17x** | 0.0008 s | 0.0010 s | **1.39x** |
| 6/8 | 1.7578 | 0.0005 s | 0.0019 s | **4.1x** | 0.0048 s | 0.0072 s | **1.52x** |
| 8/10 | 4.8828 | 0.0010 s | 0.0033 s | **3.3x** | 0.0181 s | 0.0219 s | **1.21x** |
| 10/14 | 14.9536 | 0.0041 s | 0.0102 s | **2.5x** | 0.0775 s | 0.0877 s | **1.13x** |

## What it says

1. **The out-of-core BUILD carries a fixed ~0.5-1.0 ms cost** — one HDF5 file
   creation plus the zero-fill of the dataset. Below ~0.5 MiB that fixed cost
   is the whole measurement (655x, 217x, 224x are all the same ~1 ms against
   an incore build that is a `memcpy`). Above ~2 MiB the ratio settles toward
   the ~2.5x that the actual byte traffic costs, and is still falling at
   15 MiB.
2. **The out-of-core UNFOLD costs only 1.1x-1.6x**, and the ratio *decreases*
   with size (1.13x at 15 MiB). That is a property of the port's design, not
   of HDF5: `KsymmArray::view` reads the dataset **once**, sequentially, into
   one buffer and then contracts in memory, so a whole unfold pays exactly one
   `read_slice_1d` instead of one per block. (It also keeps `hdf5-metno` — not
   built thread-safe here — off the rayon worker threads.)
3. The ratio scatter between adjacent rows (1.13x-1.62x) is larger than the
   trend; at these millisecond scales it is timer and page-cache noise, so no
   single row should be quoted as "the" number.

## The decision this measurement supports

**Keep upstream's rule: the incore/out-of-core choice stays a pure memory
test made by the caller, with no minimum-size floor in `KsymmArray`.**

* A floor cannot be justified by the unfold, which is the dominant cost in
  any real CCSD iteration and pays at most ~1.6x.
* The ~1 ms fixed build cost is only visible on arrays too small to have
  needed out-of-core in the first place — a caller reaching that branch on a
  0.001 MiB tensor has a bug in its memory estimate, not a threshold problem.
* The one situation that *would* hurt is many small out-of-core arrays created
  inside a hot loop (`kintermediates_rhf_ksymm` creates six per CCSD
  iteration). At ~1 ms each that is ~6 ms per iteration — negligible against a
  periodic CCSD iteration, but it is the number to re-check if 17-09 ever
  reports an unexplained per-iteration constant.

**Not measured, and stated as such:** nothing here exercises a tensor that
genuinely does not fit in RAM, which is the case the branch exists for. The
largest point is 15 MiB stored / 160 MiB unfolded. A real `Wvvvv` at
`nkpts^3 x nvir^4` is orders of magnitude larger and would be dominated by
disk bandwidth rather than by the fixed costs measured here. That measurement
belongs to 17-09, where a caller of that size first exists.

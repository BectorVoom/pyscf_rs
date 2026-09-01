# 17-06 SUMMARY — `pyscf/pbc/lib/ktensor.py` -> `pyscf-pbc-symm::ktensor`

**Status:** SHIPPED — all four tasks and the whole `<verification>` block.
Written INCREMENTALLY, one section per task, because the execution environment
restarts sessions every 20-40 minutes and three earlier sessions in this phase
were lost because the summary only existed at the end.

**Green:** `cargo test -p pyscf-pbc-symm --release --test ktensor` — 19/19.

Plan: `.planning/phases/17-ksymm-multigrid/17-06-PLAN.md`.
Upstream: `pyscf/pbc/lib/ktensor.py` (386 l).

---

## Task 1 — the container — DONE

**Shipped:** `crates/pyscf-pbc-symm/src/ktensor.rs` (new),
`crates/pyscf-pbc-symm/src/error.rs` (+7 additive variants),
`crates/pyscf-pbc-symm/Cargo.toml` (+`pyscf-chkfile`, +`ndarray`),
`crates/pyscf-pbc-symm/src/lib.rs` (`pub mod ktensor;`).

| upstream | port |
|---|---|
| `empty` / `empty_like` (`:26-39`) | `KsymmArray::empty` / `KsymmArray::empty_like` |
| `KsymmArray.__init__` / `_init` (`:43-81`) | `KsymmArray::empty` (incore + out-of-core branch) |
| `shape`/`ndim`/`subarray_ndim`/`subarray_shape`/`subarray_order` (`:83-107`) | same names, accessors |
| `__getitem__` / `_getitem_2d` / `_getitem_4d` (`:109-156`) | `get`/`get_2d`/`get_2d_many`/`get_4d`/`get_4d_many` |
| `__setitem__` / `_setitem_2d` / `_setitem_4d` (`:117-178`) | `set_2d_at`/`set_2d_many`/`set_4d_at`/`set_4d_many` |
| `todense` (`:180-182`) | `to_dense` |
| `fromdense` (`:184-206`) | `from_dense` — **see D-17-06-01 below** |
| `fromraw` (`:208-220`) | `from_raw`, with `to_raw` as its inverse |
| `zeros` (`:222-225`) | `zeros` |
| `_guess_input_order` (`:228-238`) | **not ported** — see the deviations |

### Structural decisions the plan required

* **`subarray_order` is an enum on the struct** (`SubarrayOrder::{C, F}`),
  not a runtime string, and it round-trips through `from_raw`. Recorded in
  the module doc: upstream's order NEVER changes a logical value —
  `fromraw` reshapes C-logically (`:217`) and `amplitudes_to_vector` ravels
  C-logically (`kccsd_rhf_ksymm.py:475`) — so the port stores every block
  row-major and carries the declared order as metadata.
* **The metadata is BORROWED, never cloned.** `KsymmMeta<'a>` holds
  `&'a KPoints`, `Option<&'a KQuartets>`, `Option<&'a MORotationMatrix>`;
  the same argument 17-CONTEXT §3.9 makes for `Symmetry` not owning a
  `Cell`, which 17-03 and 17-05 already follow. `KsymmMeta` is `Copy`, so
  `set_*` can read `self.meta.kpts` (a `'a` borrow) while holding
  `&mut self`.
* **Out-of-core goes through `pyscf_chkfile::hdf5` (D-07).**
  `grep -c hdf5-metno crates/pyscf-pbc-symm/Cargo.toml` = **0**. The
  scratch (`OutcoreStore`) is a copy of `pyscf-ao2mo::outcore`'s
  `OutcoreScratch` pattern: a uniquely-named temp `.h5`, a single flat
  `H5Complex` dataset named `data`, `Range<usize>` hyperslabs (never
  `ndarray::s![]`), RAII delete on `Drop` — upstream's `lib.H5TmpFile()`
  auto-delete.

### Deviations, recorded

1. **D-17-06-01 — upstream's `fromdense` writes to the wrong keys, both
   branches.** rank-2 (`:194-198`) passes an already-mapped IBZ index
   (`ki_ibz`) to `__setitem__`, which `set_2d` (`:244-250`) then treats as a
   FULL-BZ index and re-maps: every block whose IBZ index is not
   coincidentally also a BZ index inside the wedge is silently dropped with
   the "not in the irreducible wedge" warning. rank-4 (`:199-203`) does
   `out[m] = arr[ki,kj,ka]` with an integer `m`, which
   `index_to_coords(m, [nkpts]*3)` expands by padding the two missing axes
   with `arange(nkpts)` — `nkpts**2` coordinates for ONE value block.
   `fromdense` has **no caller and no test** in the vendored tree
   (`grep -rn fromdense pyscf/` finds only the definition and the `:386`
   alias), which is why neither bug is visible upstream. This port writes at
   the keys `set_2d`/`set_4d` actually expect, so that
   `from_dense(to_dense(x)) == x` holds bit-exactly — which is what the plan
   asks for and what upstream's version cannot satisfy.
2. **Element type is `Complex64`, not the planar `CTensor` split.** RULE 8
   bans `Complex<f64>` ACROSS THE ALGEBRA WALL; this crate never crosses it
   (no `cubecl-*` dep, no `pyscf-algebra` device call) and `kpts.rs` already
   stores every row-major complex matrix as `Vec<Complex64>` for the same
   reason. `MORotationMatrix` — whose matrices this module contracts against
   — is `Vec<Complex64>`; a planar split here would convert at every call.
3. **No `dtype` parameter.** Every `KsymmArray` upstream constructs is given
   `t1.dtype` / `t2.dtype` / `eris.fock.dtype`, all `complex128`. The
   `dtype=float` default is only reachable through
   `empty(..., metadata=None)`, which returns a plain `np.empty` — the DENSE
   path, a `Vec` in Rust, not this type.
4. **`empty` zero-fills.** `np.empty` is uninitialised (`:70`); Rust cannot
   hand out uninitialised `Complex64` safely. The only observable difference
   is that a never-written block reads as `0` instead of garbage.
5. **`_guess_input_order` (`:228-238`) is not ported.** It inspects
   `arr.flags.c_contiguous` / `f_contiguous` — a NumPy memory-layout query
   with no Rust analogue for a `&[Complex64]`, which is always contiguous.
   `SubarrayOrder` is passed explicitly instead, so the "guess" has nothing
   to guess.
6. **`NDArrayOperatorsMixin` (`:42`) is not ported.** It gives upstream
   elementwise `+`/`*` through `__array_ufunc__`; no consumer in the
   vendored tree uses it on a `KsymmArray` (they call `.todense()` first,
   e.g. `kccsd_rhf_ksymm.py:66,77`). Deferred to 17-09 if a caller needs it.

---

## Task 2 — `set_2d` / `set_4d` and the index algebra — DONE

**Shipped** (`crates/pyscf-pbc-symm/src/ktensor.rs`): `set_2d`
(`ktensor.py:240-250`), `set_4d` (`:252-264`), `index_to_coords`
(`:339-367`), `slice_to_coords` (`:369-381`), plus the `Key` / `SliceSpec` /
`Coords` vocabulary that stands in for NumPy's key objects and the
`BlockSink` / `FlatBlocks` pair that lets `set_2d`/`set_4d` write either an
in-memory buffer or the HDF5 dataset through one code path.

### The index map is tested against an INDEPENDENT dense tensor, not a round-trip

`set_4d_stores_each_triple_where_an_independent_dense_tensor_says` writes
**every** one of the `nkpts^3 = 512` full-BZ triples with a distinctive
block (`ki*1e6 + kj*1e4 + ka*100 + p`, so the triple is recoverable from any
single element), then compares the resulting store against an expectation
built straight from `kqrts.kqrts_ibz[m]` — never touching
`ktuple_to_index` or `kqrts.bz2ibz`, the two maps under test. A wrong map
puts the right numbers at the wrong slot, which a `set` -> `get` round-trip
cannot see because both directions use it; this comparison can. Same
construction one dimension down in
`set_2d_stores_each_bz_key_at_its_ibz_slot_and_discards_the_rest`.

Fixture facts printed by the tests: `si [2,2,2]` gives `nkpts = 8`,
`nkpts_ibz = 3`, `ibz2bz = [0, 6, 7]`, `len(kqrts_ibz) = 50`.

### The index arithmetic is EXHAUSTIVE, not sampled

* `slice_to_coords_is_exhaustively_numpy_arange` — **19 008 cases**: every
  `start`/`stop` in `[-2n, 2n]` plus `None`, every `step` in `[-n, n] \ {0}`
  plus `None`, for `n = 1..6`, against a reference `np.arange` written
  independently in the test. Upstream's non-clamping and negative-step
  behaviour is reproduced literally; `step == 0` is NumPy's
  `ZeroDivisionError`.
* `index_to_coords_is_exhaustive_over_a_full_generator_set_at_small_nkpts` —
  **585 cases**: the whole cross product of an 8-element generator set
  (3 integers, 4 slices, 1 index array) at every position, for key lengths
  0, 1, 2 and 3 over `shape = [3,3,3]`, against an independently written
  `lib.cartesian_prod`. It also pins the `:365-366` collapse rule: the
  `Coords::Single` variant appears for full-rank all-integer keys and for
  **nothing else**.

### `ibz2bz = [0, 6, 7]` is what makes D-17-06-01 concrete

`upstreams_fromdense_key_choice_would_drop_two_of_three_blocks_here` shows
the arithmetic: upstream's `fromdense` passes the IBZ indices `{0, 1, 2}` to
`set_2d`, which tests them against `ibz2bz = [0, 6, 7]`; only `0` survives,
so upstream would keep 1 of 3 blocks and warn away the other 2. The test
asserts that, so the deviation cannot be "simplified back" to upstream's
line without a failure.

---

## Task 3 — `transform_2d` / `transform_4d` — DONE

**Shipped** (`crates/pyscf-pbc-symm/src/ktensor.rs`): `transform_2d`
(`ktensor.py:266-287`), `transform_4d` (`:289-337`), `rot_of`
(`getattr(rmat, pi*2)[k][iop]`), the `Blocks` read view, and the host-only
`zgemm` / `transpose` / `transpose_102` / `transpose_021` helpers.
`transform_4d` performs the four contractions in **upstream's own order**
(`:322-336`), not as one fused einsum, so the operation count and the
summation order match line for line.

### Every `(label, trans)` combination is tested individually

The plan's trap, and `14-VERIFICATION`'s twice-recorded defect class. Both
comparisons are against a **direct summation written out in the test file**
— `sum_{ij} A[ij] ri[ik] rj[jl]` and
`sum_{ijab} A[ijab] ri[ik] rj[jl] ra[ac] rb[bd]` — never a second call into
the port:

| | combinations | worst residual | tol |
|---|---:|---:|---:|
| `transform_2d` (2 labels x 2 positions, 2 conj x 2 positions) | **16** | 4.965e-16 | 1e-13 |
| `transform_4d` (2^4 labels x 2^4 conj) | **256** | 1.343e-15 | 1e-13 |

Both tests `assert_eq!` on the combination COUNT (16 / 256), so a future
edit cannot quietly shrink the enumeration. `nocc = 2`, `nvir = 3` so the
`o`/`v` axes have different dimensions and a swapped label is a shape error,
not a silent wrong answer.

### Oracle-free invariants

* `transform_with_an_identity_rotation_returns_the_block_bit_exactly` — at a
  k-point (and a triple) that takes the CONTRACTION branch, not the
  `ki == ki_ibz_bz` shortcut, with an identity `MORotationMatrix`: every
  `trans` must give the stored block back **bit-for-bit** (`to_bits()`), for
  all 4 rank-2 and all 16 rank-4 `trans` combinations.
* `unfolding_the_whole_ibz_and_refolding_reproduces_the_stored_blocks_bit_exactly`
  — unfold all `nkpts` (rank 2) / `nkpts^3` (rank 4) keys, feed them all back
  through `set_*`, compare every stored block bit-for-bit.
* `hermiticity_survives_exactly_the_mixed_trans_combinations` — for a square
  label and a Hermitian block, `trans` `'nc'`/`'cn'` preserve Hermiticity
  (measured **7.850e-17 / 1.114e-16**, gated 1e-14) and `'nn'`/`'cc'` do
  **not** (measured **1.192 / 1.735**, asserted `> 1e-3`). Both halves are
  asserted, with the failure message saying that a small value in the second
  group would mean the fixture stopped exercising a genuinely complex
  rotation — not that asserting Hermiticity everywhere became valid. This is
  the 17-05 `mo_coeff_elementwise_comparison_is_not_a_valid_gate` pattern.
  **The plan's phrasing "for a Hermitian input the output is Hermitian, for
  every op and both `trans` values" is wrong as written** and is corrected
  here: `R^T A R` and `R^H A R^*` are not Hermitian for a complex unitary
  `R`, and the test proves it rather than asserting an identity that does
  not hold.

### Speed — the unfold is a rayon `par_iter`, and it is bit-identical

`get_2d_many` / `get_4d_many` (and therefore `to_dense`) are
`par_iter().map().collect()` over the requested keys: each key writes exactly
one output slot and reads a shared immutable `Blocks` view, so it is
disjoint by construction with no reduction — the same argument 17-05's star
unfolds make, and hence no `oracle_sum` ordering to protect. `collect()`
restores key order.

`the_unfold_is_bit_identical_at_1_and_8_rayon_workers` varies the worker
count inside ONE process with explicit `rayon::ThreadPool`s (17-05's
strictly-stronger-than-`RAYON_NUM_THREADS` pattern) and compares all
**512** unfolded rank-4 blocks by `to_bits()`.

For the out-of-core store, `view()` reads the HDF5 dataset **once**,
sequentially, before the parallel section — both because it is faster (see
the measurement) and because `hdf5-metno` is not built thread-safe here, so
no worker thread may touch it.

---

## Task 1 speed — the incore / out-of-core crossover, MEASURED

Full table and reasoning: `.planning/phases/17-ksymm-multigrid/17-06-MEASUREMENT.md`.
Reproduce with
`cargo test -p pyscf-pbc-symm --release --test ktensor -- --nocapture measure_the_incore_outcore_crossover`.

**First finding: `ktensor.py:54-83` has no heuristic to span.** `incore`
comes straight off the metadata dict (`:50`) and every caller decides with a
pure memory test (`_memory_4d(...) + lib.current_memory()[0] < cc.max_memory
* .9`, `kccsd_rhf_ksymm.py:153-166`, `:218-235`, `:289-323`, `:413-417`).
There is no size constant anywhere to copy — so the measurement answers
"what does forcing out-of-core COST" instead, which is the question that
decides whether the port needs a floor.

`si [2,2,2]`, 50 stored rank-4 blocks, 512-block unfold, both paths asserted
bit-identical at every size:

| `nocc`/`nvir` | stored MiB | `from_raw` in/out | ratio | unfold in/out | ratio |
|---|---:|---|---:|---|---:|
| 1/1 | 0.0008 | 0.0000 s / 0.0010 s | 655x | 0.0001 s / 0.0001 s | 1.24x |
| 2/2 | 0.0122 | 0.0000 s / 0.0005 s | 217x | 0.0001 s / 0.0002 s | 1.62x |
| 3/4 | 0.1099 | 0.0000 s / 0.0009 s | 224x | 0.0003 s / 0.0004 s | 1.24x |
| 4/6 | 0.4395 | 0.0001 s / 0.0009 s | 17x | 0.0008 s / 0.0010 s | 1.39x |
| 6/8 | 1.7578 | 0.0005 s / 0.0019 s | 4.1x | 0.0048 s / 0.0072 s | 1.52x |
| 8/10 | 4.8828 | 0.0010 s / 0.0033 s | 3.3x | 0.0181 s / 0.0219 s | 1.21x |
| 10/14 | 14.9536 | 0.0041 s / 0.0102 s | 2.5x | 0.0775 s / 0.0877 s | 1.13x |

**Conclusion — no minimum-size floor is added.** The out-of-core BUILD
carries a fixed ~0.5-1.0 ms HDF5 file-creation cost, which is the entire
"655x" at 0.0008 MiB and decays to ~2.5x by 15 MiB. The out-of-core UNFOLD
— the cost that dominates any real CCSD iteration — is only 1.1x-1.6x and
*falling* with size, because `view()` reads the dataset once instead of once
per block. The choice therefore stays upstream's caller-side memory test.
The one number to re-check later: `kintermediates_rhf_ksymm` builds six
arrays per CCSD iteration, so ~6 ms per iteration of fixed cost if they all
go out of core — negligible, but it is the constant to look for if 17-09
reports an unexplained per-iteration overhead.

**Explicitly NOT measured:** a tensor that genuinely does not fit in RAM —
the case the branch exists for. The largest point here is 15 MiB stored /
160 MiB unfolded; a real `Wvvvv` is orders of magnitude larger and would be
disk-bandwidth-bound rather than fixed-cost-bound. That measurement belongs
to 17-09, where a caller of that size first exists.

---

## Task 4 — the acceptance test uses a REAL converged SCF — DONE, with a
## documented substitution

**The plan's Task 4 asks for "a real `khf_ksymm` Fock store". 17-07
(`khf_ksymm`) has not shipped — it is the next plan — so a `khf_ksymm` Fock
store does not exist to test against, and this plan does NOT fabricate one.**
What it uses instead is the quantity upstream's own `KsymmArray` callers
store, computed from a real converged SCF:

```
kintermediates_rhf_ksymm.py:26-33   Fki, label 'oo', trans 'cn',
                                    Fki[ki] = eris.fock[ki,:nocc,:nocc]
                        :47-56      Fkc, label 'ov', trans 'cn'
                        :71-80      Fac, label 'vv', trans 'cn'
```

`acceptance_real_converged_krhf_mo_blocks_round_trip_through_ksymmarray`
converges `si [2,2,2]` KRHF/FFTDF at `precision = 1e-10`,
`conv_tol_grad = 1e-10`, `conv_tol = 1e-11` (`nocc = 4`, `nvir = 4`,
`nkpts = 8`, `nkpts_ibz = 3`), builds a REAL `MORotationMatrix` from the
converged MOs and overlaps (17-05's `MORotationMatrix::build`), forms four
MO-basis one-electron blocks, stores each as a `KsymmArray` over the IBZ,
reads back **every** BZ k-point through `to_dense`, and compares to the
dense full-BZ array.

The transform law it pins is the one upstream's `('oo'|'ov'|'vv', 'cn')`
encodes:

```
A[k2] = rot^H A[k1] rot ,   rot = C[k1]^H S[k1] R^H C[k2]
```

which holds for any operator commuting with the space group.

### Measured, max over every k and every element

| quantity | `trans = 'cn'` (upstream's) | `trans = 'nc'` (the control) |
|---|---:|---:|
| `C_o^H F C_o` — upstream's `eris.fock` oo block | **1.186e-13** | 1.186e-13 (identical) |
| `C_o^H h C_o` | **5.455e-13** | 5.455e-13 (identical) |
| `C_o^H h C_v` | **3.003e-12** | **1.518e-6** |
| `C_v^H h C_v` | **3.607e-12** | 3.607e-12 (identical) |

**Worst `'cn'` residual: 3.607e-12** — gated at `GATE_B_TOL = 1e-9`, which is
17-01 Task 2's measured Gate-B floor (4.481e-10 at default `cell.precision`).
280x of margin. If it ever fails, tighten the fixture, not the gate.

### The finding worth recording: three of the four blocks CANNOT discriminate the `trans` flag, and why

`'cn'` gives `R^H A R`, `'nc'` gives `R^T A R*`. For a REAL `A` the second is
the conjugate of the first, so when the result is also real the two coincide
*identically*. The test measures the block structure over every k and prints
it:

| block | max abs off-diagonal | max abs imaginary part |
|---|---:|---:|
| `C_o^H F C_o` | 1.636e-13 | 5.621e-17 |
| `C_o^H h C_o` | 2.946e-13 | 1.317e-16 |
| `C_v^H h C_v` | 9.026e-13 | 4.615e-16 |
| `C_o^H h C_v` | (rectangular) | **7.582e-7** |

All three square blocks are real and DIAGONAL to machine precision. That is
not an accident and not a defect: **Schur's lemma.** At each k-point of `si`'s
`[2,2,2]` mesh every occupied (and every virtual) irrep of the little co-group
appears exactly once, and an operator commuting with the group restricted to a
single copy of an irrep is a multiple of the identity. So `C^H h C` and
`C^H F C` are both diagonal, both real, and both blind to the antiunitary
convention. The rectangular `C_o^H h C_v` block is not, and it discriminates
by a factor of **5 x 10^5** (3.003e-12 vs 1.518e-6).

**Consequence, stated so 17-07/17-09 do not have to rediscover it:** a
symmetric cell's canonical MO one-electron blocks are a weak probe of the
`trans` flag. The strong probe is the synthetic 256-combination enumeration in
`transform_4d_matches_an_independent_einsum_for_every_label_and_trans`,
which uses genuinely complex unitaries and a directly written-out einsum. The
real-data test proves the container against a quantity that exists today; the
synthetic test proves the antiunitary algebra. Neither replaces the other.

The test asserts BOTH halves: `'cn' < 1e-9` for all four, and — written from
the measurement, not guessed — that the wrong `trans` is distinguishable
whenever the reference blocks carry an imaginary part.

---

## Verification — all of `17-06-PLAN.md`'s `<verification>` block

| requirement | result |
|---|---|
| `cargo test -p pyscf-pbc-symm --release` green | `ktensor` **19/19** (120 s), `geom` 7/7, `group` 8/8, `space_group` 15/15, `symmetry` 21/21, `kpts_ibz` 5/5, `kpts_ktuples` 8/8, `kpts_transform` **14/14** (496 s — re-run, since this plan touched the neighbouring `error.rs`). `tests/basis.rs` (17-04's, ~5200 s) was NOT re-run: its inputs are unchanged and `error.rs` gained only additive variants — the same call 17-05 made and recorded. |
| 4-d index map tested against an independently built dense tensor, not a round-trip | `set_4d_stores_each_triple_where_an_independent_dense_tensor_says` — all 512 triples written, expectation built from `kqrts.kqrts_ibz` alone, compared BIT-for-bit. |
| Every `(label, trans)` combination has its own test | **16** for `transform_2d`, **256** for `transform_4d`, each against a directly written-out einsum; the tests `assert_eq!` on the counts so the enumeration cannot be quietly shrunk. Worst residuals 4.965e-16 and 1.343e-15 against a 1e-13 tolerance. |
| Out-of-core path goes through `pyscf_chkfile::hdf5`; `grep hdf5-metno crates/pyscf-pbc-symm/Cargo.toml` empty (D-07) | `grep -c hdf5-metno` = **0**. `OutcoreStore` uses `pyscf_chkfile::hdf5` + `pyscf_chkfile::H5Complex`, RAII-deleted temp file. |
| Incore and out-of-core give identical results | `incore_and_outcore_give_identical_results` — all 512 unfolded blocks and the whole `to_dense` compared by `to_bits()`, plus a write-through-HDF5 check that `set_4d_many` lands in the same slots. |
| Unfold bit-identical at 1 and 8 rayon workers | `the_unfold_is_bit_identical_at_1_and_8_rayon_workers`, explicit `rayon::ThreadPool`s inside ONE process, 512 blocks by `to_bits()`. |
| No `mod tests` in any `src/*.rs` | `grep -rn "mod tests" crates/pyscf-pbc-symm/src/` — no matches (exit 1). |
| clippy | `cargo clippy -p pyscf-pbc-symm --all-targets` reports **nothing** in `src/ktensor.rs` or `tests/ktensor.rs`. (Pre-existing warnings remain in `src/basis.rs` / `src/space_group.rs` / `tests/basis.rs` — not this plan's.) |
| incore/out-of-core crossover MEASURED, not guessed | `17-06-MEASUREMENT.md`. Headline: upstream has no size constant to copy (its callers use a pure `max_memory` test); the out-of-core unfold costs only 1.1x-1.6x and falling, the out-of-core build carries a fixed ~1 ms; **no minimum-size floor added.** |

### Environment note

The sibling `libxc_rs` workspace was being regenerated by another agent during
this session, so `cargo` intermittently failed with
`failed to get libxc-rkernel-mgga_* as a dependency of libxc-reval` /
`no targets specified in the manifest`. Every such failure was transient and
every test above was re-run to green. It is recorded only so a later reader
does not mistake it for a defect in this plan.

---

## What 17-07 (and 17-09) must add

1. ~~**17-07 — the acceptance test the plan actually named.**~~ **CLOSED
   2026-09-02** by `crates/pyscf-pbc-symm/tests/ktensor_ksymm_scf.rs`
   (`ksymm_scf_fock_store_unfolds_through_ksymmarray`), once 17-07's
   `KsymAdaptedKrhf` existed. The store is filled from the ksymm SCF's **own
   IBZ-length output** via `set_2d_at` at the irreducible representatives, and
   every full-BZ k-point is read back with `get_2d`.

   **The comparison had to be chosen carefully.** Comparing against MO blocks
   from a separate full-BZ KRHF is NOT sound: orbitals are defined only up to a
   unitary rotation inside each degenerate subspace, so the two SCFs' `C` need
   not agree and `C^H h C` then legitimately differs (17-CONTEXT §3.1; this
   plan met the same wall from the other side, where Schur's lemma blinded
   three of its four blocks). Both sides therefore start from **one** ksymm
   SCF's orbitals, and the test compares two independent unfolds:
   `KsymmArray::get_2d` (this plan's `transform_2d` + `MORotationMatrix`)
   against projecting with the orbitals `KPoints::transform_mo_coeff` unfolds
   (17-05, independently gated). They share no code below `KPoints`.

   Measured on `si [2,2,2]`, `nkpts_ibz = 3`, `nocc = nvir = 4`, trans `'cn'`:

   | block | max &#124;get_2d − project(transform_mo_coeff)&#124; |
   |---|---|
   | `oo` | 8.255e-14 |
   | `ov` | 3.842e-13 |
   | `vv` | 3.318e-12 |

   Worst 3.318e-12, against the 1e-9 Gate-B floor. `hcore` is used rather than
   the Fock because the MO-basis Fock is diagonal by construction and could
   not see a wrong rotation — the same reason this plan gated the `hcore`
   blocks alongside the Fock one.

   This plan's stand-in
   `acceptance_real_converged_krhf_mo_blocks_round_trip_through_ksymmarray`
   is **kept**, as this section asked: it pins the algebra independently of
   `khf_ksymm`'s bookkeeping.

   Superseded original text: Once
   `khf_ksymm` has a Fock store, add a test that builds it as a
   `KsymmArray` over the IBZ (label `'oo'`, trans `'cn'` — or whatever
   `khf_ksymm` declares) directly from the SCF's own container rather than
   from a test-local projection, reads back every BZ k-point, and compares to
   the dense full-BZ Fock at `GATE_B_TOL`. This plan's
   `acceptance_real_converged_krhf_mo_blocks_round_trip_through_ksymmarray`
   is the stand-in and should be kept, not replaced: it pins the algebra on a
   quantity independent of `khf_ksymm`'s own bookkeeping.
2. **17-09 — a rank-4 test on REAL data.** Every rank-4 assertion here is
   either an index-map comparison against a synthetic dense tensor or an
   einsum comparison against a synthetic rotation. The first real rank-4
   quantity is `kmp2_ksymm`/`kccsd_rhf_ksymm`'s MO-basis `eris.oovv`, which
   needs the periodic AO->MO transform; when it exists, store it as a
   `KsymmArray` (label `'oovv'`, trans `'ccnn'`) and compare the unfolded
   `nkpts^3` blocks to the dense ERIs.
3. **17-09 — `make_k4_ibz("s2")`/`("s4")`.** 17-05 shipped `"s1"` only;
   `KQuartets` (and therefore every rank-4 `KsymmArray`) is built on `"s1"`.
   Nothing here is blocked, but a `"s2"`/`"s4"` `KQuartets` changes
   `kqrts_ibz` and so re-opens every index-map assertion in this file — they
   must be re-run against the new tables, not assumed.
4. **17-09 — the out-of-core measurement at a size that does not fit.**
   `17-06-MEASUREMENT.md` explicitly does NOT measure the case the branch
   exists for; the largest point is 15 MiB stored. A real `Wvvvv` is
   disk-bandwidth-bound and needs its own number.
5. **`NDArrayOperatorsMixin`** (elementwise `+`/`*` on a `KsymmArray`,
   `ktensor.py:42`) is not ported — no vendored consumer uses it. If 17-09
   wants `t2new += ...` on the container rather than on `todense()`, that is
   where it lands.

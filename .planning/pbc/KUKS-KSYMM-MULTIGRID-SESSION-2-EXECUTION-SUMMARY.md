# KUKS + k-symmetry + multigrid — session 2 execution record

**Plan:** [`KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md`](./KUKS-KSYMM-MULTIGRID-OPTIMISATION-PLAN-2.md)
**Implementation landed:** 2026-09-03 → 2026-09-04 (a prior session)
**Verification, defect hunt and this record:** 2026-09-04
**Machine:** 16 cores, 30 GiB RAM, CubeCL **CPU** runtime
(`pyscf-algebra` `default = ["cpu"]`; the ROCm iGPU has no f64, so
`PYSCF_BACKEND=rocm` resolves to CPU — every number below is a CPU-runtime
number and every GPU claim stays UNVERIFIED, per the plan's backend note).

---

## 0. What this session actually did

The plan's **code** was already fully landed when this session started; what
had never been done was **§5 verification**. Running it found a real defect
that had been shipped green-looking, and the bulk of this session went into
diagnosing and fixing it.

| | |
|---|---|
| Items whose code was found already landed | P-10 guard, A-00, A-01, A-02, S-06, S-03, M-01, M-06, M-07, M-08, M-09, M-10, M-04 step 3 |
| Items completed this session | **S-04** (audit + the hoist it found), **GATE B repair** (M-06 defect below) |
| Items still owed | every **measurement**: P-10 baselines, A-00's stage table, M-01's instrument rows, S-02 step 4's ratio and default flip |

---

## 1. DEFECT FOUND AND FIXED — M-06's resident handles were thread-bound

**Severity: high.** GATE B (thread-count bit-identity, D-PBC-17) was **RED** on
the v2 multigrid path. It had never been run against the landed code.

### Symptom

```
thread 'DSD-0-0' panicked at cubecl-runtime-0.10.0/src/memory_management/memory_pool/sliced_pool.rs:47:36:
index out of bounds: the len is 0 but the index is 0
thread '<unnamed>' panicked at cubecl-runtime-0.10.0/src/client.rs:105:14:
called `Result::unwrap()` on an `Err` value: CallError
```

Failing: `v2_get_j_is_bit_identical_across_thread_counts`,
`v2_nr_rks_is_bit_identical_across_thread_counts` (`multigrid_threads.rs`),
`eval_rho_g_is_bit_identical_across_thread_counts_v2` (`multigrid2.rs`).

### It is NOT the environment

The first observation was made while the box was at 29/30 GiB and the harness
was killing tasks for memory, so memory exhaustion was the obvious suspect. It
was **wrong**: the failure reproduces deterministically on an idle machine
(25 GiB free, load 0.67), in 15 s, standalone, outside cargo.

### Root cause (VERIFIED by reading cubecl, not inferred)

`cubecl-common-0.10.0/src/stream_id.rs`:

```rust
pub struct StreamId {
    /// The value representing the thread id.
    pub value: u64,
}
#[cfg(multi_threading)]
std::thread_local! { static ID: RefCell<Option<u64>> = ... }
```

A `StreamId` **is the OS thread id**, `ComputeClient::load` sets
`stream_id: None` so every call resolves `StreamId::current()`, and CubeCL
partitions its memory pools per stream. `SlicedPool::find` (`:47`) resolves a
binding by indexing `self.pages[descriptor.page()]` **of the current stream's
pool** — so a `Handle` is only resolvable on the stream that allocated it.

M-06 cached `PairSlotBatchDevice` (eleven geometry `Handle`s + output handles)
in a `OnceLock` on `BatchedLevel`, alive for the whole SCF, while the drivers
above run under changing rayon pools. On the next call the handles were
re-resolved on a **fresh** stream whose pool has zero pages → the panic.

### The natural experiment that proved it

| test | switches rayon pool | caches resident handles | result |
|---|---|---|---|
| `multigrid_batch` (resident vs streamed, 4 tests) | no | **yes** | PASS |
| `gate_e_*_v2`, `int_rho_matches_tr_dm_s_v2` | no | **yes** | PASS |
| `v1_*` thread-identity | **yes** | no | PASS |
| `v2_get_j`, `v2_nr_rks`, `eval_rho_g_v2` | **yes** | **yes** | **PANIC** |

The panic needs *both* conditions. Neither alone reproduces it.

### Fix

`crates/pyscf-kernels/src/multigrid_pair.rs` — record the allocating stream in
`PairSlotBatchDevice` and replay every launch and read onto it with
`StreamId::executes`, the API CubeCL provides for exactly this:

```rust
stream: StreamId,                    // captured in `new()`, before the uploads
...
Ok(self.stream.executes(|| dispatch_backend!(client, c, Rt, { launch_… })))
```

applied to all four entry points (`rho`, `integrate`, `rho2`, `integrate2`).

**Bit-parity: EXACT.** Pinning moves no arithmetic — same kernel, same lanes,
same order; only which stream the work is queued on changes.

### Verification

`cargo test -p pyscf-pbc-dft --test multigrid_threads` →
**`ok. 4 passed; 0 failed`** (132.50 s). Was 2 passed / 2 failed.

### Scope check — A-02 is NOT affected

`AoBlockDevice` (A-02) caches handles too, but `eval_ao_kpts_with_images`'s
image loop is serial (no `par_iter`) and the handle is created and consumed
inside one iteration, so it never outlives its creating thread. A-00's gate
(`eval_ao_stages`) passes.

### Prevention

Any type in `pyscf-kernels` that stores a CubeCL `Handle` **beyond the call
that created it** must also store the `StreamId` it was created on and replay
onto it. `AoKAccumulator` and `PairSlotBatchDevice` are the two such types
today; `PairSlotBatchDevice` now does, `AoKAccumulator` does not need to
(single-threaded loop) but would if that loop were ever parallelised.

---

## 2. S-04 — J/K pair-invariant audit (CLOSED, with a finding)

The item was written as "audit only; close with the table if nothing is
found". Something was found.

### The audit table — `get_k_kpts_opts`'s `k1` loop

| work in the `k1` loop | pair-invariant? | status |
|---|---|---|
| `ao1t = ao1_kpts.at(k1)` | no (depends on `k1`) | correct |
| `ao_dms` (`dm . conj(ao2T)`) | per `k2` | already hoisted above the `k1` loop, and outside the `nset` loop |
| `vr_dm` allocation | yes | already hoisted (U-06 step 5) |
| `gv` | yes | already hoisted above both loops |
| `coulG(dk)` / `expmikr(dk)` construction | per `dk` | already cached on `Fftdf` (W-01) |
| **`coulG` / `expmikr` COPY out of that cache** | **yes** | **FOUND — fixed below** |
| `build_rho1`, FFT, `coulG` multiply, iFFT, `contract_vr_aodm`, phase, `accumulate_vk` | no | genuinely per pair |

### The finding

`Fftdf::coulg_and_expmikr` returns `Arc<(Vec<f64>, Option<CTensor>)>` — W-01
built that cache precisely so the tables are not rebuilt per pair. Both call
sites then did:

```rust
let (coulg, expmikr) = { let entry = df.coulg_and_expmikr(..)?; (entry.0.clone(), entry.1.clone()) };
```

`entry.0.clone()` deep-copies `ngrids` f64 and `entry.1.clone()` deep-copies
two more `ngrids` planes — **3·ngrids doubles per k-pair**. At the gate mesh
(`ngrids = 29 791`) that is 715 KiB per pair, ~45 MiB per `get_k_kpts` at 64
pairs, ×`nset`. The cache saved the *construction* and then paid for the
*copy* — the same defect shape M-06 had (cache the build, still move the
bytes).

### Fix

`crates/pyscf-pbc-df/src/fft_jk.rs`, both sites (`get_k_kpts_opts:443` and
`kk_symmetric_pair_loop:785`) — hold the `Arc` and borrow:

```rust
let entry = df.coulg_and_expmikr(dk, omega, inner_exxdiv, kpts, &gv)?;
let (coulg, expmikr) = (&entry.0, entry.1.as_ref());
```

**Bit-parity: EXACT** — the same bytes, read in place instead of copied.
**RULE T bytes removed:** `3 · ngrids · 8` B per k-pair per set.

---

## 3. Gate status

| gate | status | evidence |
|---|---|---|
| **§5.7 lints** | **PASS** | `check-dependency-wall`: cubecl containment intact (ALG-06); `check-orphan-modules`: 336 files all reachable; `check-no-fma`: 7 asm files, no FMA |
| **GATE E (kernels)** | **PASS 22/22** | `multigrid_pair` 4, `pbc_eval_ao_k` 3, `eval_gto_oracle` 8, `eval_gto_lge1` 7 |
| **GATE AO (A-00)** | **PASS** | `eval_ao_stages`: thread bit-identity + screen inside the W-09 gate |
| **GATE B** | **PASS 4/4 after the §1 fix** | was 2/4; `multigrid_threads` 132.50 s |
| `multigrid_batch` (M-06/M-07/M-08) | **PASS 4/4** | resident vs streamed bit-identical, incl. the register-bound streaming fallback |
| GATE E (dft), GATE MG-SCF, GATE C, S-03's test, GATE A / GATE U | **not yet reported** | re-run in flight at the time of writing; **not claimed green** |

---

## 4. Not done, and why

* **Every measurement the plan asks for.** P-10's idle baselines, A-00's
  four-stage attribution table (Q9), M-01's per-level instrument rows (Q12,
  Q13), M-07's chunk-count table (Q11), S-02 step 4's `Band` vs `Reference`
  ratio (Q15) and the default flip, S-03's `× N_ibz/N` check (Q14). The GATE S
  ledger therefore has **no new rows**, and `baselines/` still ends at
  2026-09-02.
* **Why not:** the session's whole budget went to §5 verification and the M-06
  defect. RULE O forbids quoting a ratio off a loaded machine, and for most of
  the session the box was at load 18-25 with an unrelated `libxc_rs` build and
  an `xcqual-baseline` benchmark running — neither of them this session's to
  kill. P-10's own guard refuses to write a baseline above load 4.0, by design.
* **Consequence:** the plan's speed claims for A-01, A-02, M-06, M-07, M-08,
  M-10 remain **UNMEASURED**. They are asserted bit-exact and now (after §1)
  actually pass their bit-exactness gates, but nothing in this repository yet
  shows they made anything faster.

## 5. What the next session should do, in order

1. Finish the §5 re-run in flight and record GATE A / GATE U / GATE C / GATE E
   (dft) / GATE MG-SCF.
2. **P-10 on a genuinely idle box** — it gates every S- ratio and the S-02 flip.
3. A-00's stage table (Q9), then M-01's instrument (Q11-Q13).
4. S-02 step 4: measure, then flip `JkRoute` only where the ratio is > 1.0.
5. Only then quote any speed number from this plan.

## 6. Erratum for AGENTS.md §4

AGENTS.md §4 mandates reading
`/home/user/Documents/workspace/cubecl_manual/manual/cubecl_error_guideline.md`
before touching code on any cubecl failure. **That file does not exist.** The
directory that does exist,
`cubecl_manual/manual/Cubecl/cubecl_error_solution_guide/`, holds two documents
(`mismatched types.md`, `calling a "normal" Rust function from inside a cube
macro function fails in CubeCL.md`), neither covering runtime or memory-pool
failures. The §1 diagnosis therefore proceeded from evidence — a reproducer, a
natural experiment, and the CubeCL source — rather than from the guideline. The
path in AGENTS.md should be corrected, and the §1 stream/`Handle` lifetime rule
is a candidate entry for that guide.

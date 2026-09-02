# Phase 16 — plan review + speed & memory optimisation pass

**Written:** 2026-09-02, before any Phase-16 code.
**Scope:** `PBC-MASTER-PLAN §8.8`'s ten-plan Phase-16 table and its Reuse note,
reviewed against the vendored PySCF 2.12.1 tree and the current Rust workspace;
then a speed-and-memory pass over the fourteen-plan replacement in
`16-CONTEXT §4`.
**Outcome:** 7 defects in `§8.8` (all fixed in the new plan set, 4 of them
scope-level), 5 speed/memory rulings folded into the plans as
**D-PBC-29**, 3 findings recorded but deliberately not acted on.

Every claim carries the file and line that proves it. **Nothing here is a
measurement of new code** — no Rust was written or run. The numbers below are
*derived* byte counts and flop ratios, and 16-01 measures the ones that can be
measured before implementation starts.

---

## 1. What `§8.8` got right, so it does not get lost

* **The line count is exact.** "13,675 + 852 lines upstream" reproduces
  `wc -l` on `pyscf/pbc/cc/*.py` and `pyscf/pbc/ci/*.py` to the line, and the
  ten-plan table accounts for 12,276 of the 13,675 — the balance being the
  1,071 `*_ksymm` lines correctly assigned to Phase 17's 17-09, plus
  `kccsd_t_rhf_slow.py` (271) and `__init__.py` (57). Compare Phase 15, where
  three of five plans were wrong about the starting state: `§8.8`'s inventory
  arithmetic is the best in the master plan.
* **"This is the largest phase"** is right, and the Reuse note is right about
  *why* — `Wvvvv` at `nkpts³ × nvir⁴` really is the thing that decides whether
  the phase runs at all. §2.3 below turns that instinct into numbers.
* The eight-way split into `kintermediates` / `KCCSD` / `(T)` / `EOM` / `CI` /
  `rdm` follows upstream's own module boundaries, so no plan straddles two
  files and every plan has a natural test surface.

---

## 2. Memory — the arena, and three defects Phase 16 would inherit

### 2.1 The arena `§8.8` tells Phase 16 to reuse cannot hold a Phase-16 tensor

`§8.8`'s Reuse note: *"`pyscf-ccsd` already owns the molecular `WorkspacePool`
tensor arena and `PYSCF_MAX_MEMORY` pre-flight refusal. `pyscf-pbc-cc` MUST
reuse both."*

Three things are wrong with that sentence, and the third is disqualifying:

1. **It is not in `pyscf-ccsd`.** It is
   `crates/pyscf-runtime/src/workspace_pool.rs`; `pyscf-ccsd` is a *consumer*
   (`rdm.rs:32`, `lambda.rs:37`). Phase 16 must therefore modify
   `pyscf-runtime`, a crate below the entire workspace — a materially
   different change from adding a module to `pyscf-ccsd`.
2. **`pyscf-ccsd` has no complex arithmetic at all.**
   `grep -c "Complex64\|Complex<f64>\|c64" crates/pyscf-ccsd/src/*.rs` is zero
   on every file. Reuse of the molecular *code* is therefore limited to the
   kernel/DIIS **shape**; none of the arithmetic transfers.
3. **The pool is f64-typed all the way down.** `shape_bytes(shape)` is
   `shape.iter().product() * 8` (`workspace_pool.rs:278-280`),
   `TensorBackend::InMemory(Box<[f64]>)`, `as_slice -> Vec<f64>` (`:397`),
   `write_slice(&[f64])` (`:422`), `with_mut_slice(&mut [f64])` (`:461`).
   Every k-point CC tensor is `complex128` (`kccsd_rhf.py:553-554`).

**Why not just reinterpret `Box<[f64]>` as complex pairs.** Because
`shape_bytes`'s `* 8` feeds `try_reserve`, and `try_reserve` is the HARD
`MemoryLimitExceeded` refusal (`:266-274`). A complex tensor whose size is
computed with `* 8` reports **half** its true footprint to the one mechanism
whose job is to refuse before an OOM. On the machine this project actually runs
on — where 17-12's entire host suite is SIGKILLed with exit 137 (`STATE.md`) —
a 2× under-report is not a rounding error, it is the failure mode.

**Ruled (D-PBC-29 clause 1, 16-02 Task 1):** a distinct `ZWorkspacePool` with
`shape_bytes = product * 16`, sharing the budget accounting, the free-list and
the `InMemory | Spilled` split. `WorkspacePool`'s f64 API is left byte-for-byte
unchanged, and 16-02's verification proves it by an empty diff plus a green
`cargo test -p pyscf-ccsd`.

### 2.2 `as_slice` copies the whole tensor, and `with_mut_slice` holds a global lock

Two properties of the f64 pool that are harmless for molecular CCSD and
pathological at `nkpts³`:

* **`as_slice(&self, id) -> Result<Vec<f64>, _>`** (`:397-410`) — for an
  `InMemory` backend it is `Ok(b.to_vec())`, a **full copy on every access**.
  Molecular CCSD touches a workspace buffer a handful of times per iteration.
  A k-point `update_amps` touches ERI blocks inside an `nkpts³` loop. Copying
  `Wvvvv` per access is not a constant factor, it is a different complexity
  class.
* **`with_mut_slice`** (`:461-483`) takes the pool's single `Mutex<Vec<..>>`
  and **runs the caller's closure while holding it** (`:463-466`, and the doc
  comment says so). Two rayon threads working on two *different* buffers
  serialise. Every contraction in this phase is a rayon loop over k-triples,
  so this would cap the whole phase at one core.

Neither is a defect *in* `WorkspacePool` — for its shipped callers the
semantics are fine and its `heap_alloc_count.rs` test pins the property that
actually mattered there. They are defects in the instruction to reuse it.

**Ruled (D-PBC-29 clause 1, 16-02 Tasks 1 and 3):** `ZWorkspacePool` and
`KTensor` provide (a) a block accessor that borrows rather than copies, and
(b) per-buffer rather than per-pool locking. 16-02 test 12 asserts two threads
writing two different blocks make concurrent progress, so a lock-shaped
regression fails a test instead of silently costing 15 cores.

### 2.3 The storage tiers — and why the gate must cross one

`kccsd_rhf.py` selects between **three** storage tiers in four separate places
(`:132-137` `Woooo`, `:179-192` `Wvoov`/`Wvovo`, `:423-455` `Wvvvv`,
`:777-832` the seven `_ERIS` blocks): skip entirely / incore `np.empty` / HDF5
`create_dataset`.

Derived footprints, complex128 at 16 B/element, `nkpts³ · <block>`:

| cell / basis | mesh | `oooo` | `oovv` | `vovv` | **`vvvv`** | 7 blocks |
|---|---|---|---|---|---|---|
| diamond `gth-szv` (nocc 4, nvir 4) | 2×2×2 | 2.0 MiB | 2.0 MiB | 2.0 MiB | **2.0 MiB** | 14 MiB |
| diamond `gth-szv` | 3×3×3 | 77 MiB | 77 MiB | 77 MiB | **77 MiB** | 538 MiB |
| diamond `gth-szv` | 4×4×4 | 1.0 GiB | 1.0 GiB | 1.0 GiB | **1.0 GiB** | 7.0 GiB |
| diamond `gth-dzvp` (nocc 4, nvir 22) | 2×2×2 | 2.0 MiB | 60 MiB | 333 MiB | **1.79 GiB** | 2.26 GiB |
| diamond `gth-dzvp` | 3×3×3 | 77 MiB | 2.3 GiB | 12.5 GiB | **68.7 GiB** | 87 GiB |

Two consequences the plans now carry:

* **Every `§9.2` reference cell is `gth-szv`** (`test_systems.rs:8-14`), where
  `vvvv` at 2×2×2 is **2 MiB**. A phase gated only on those fixtures would
  ship the incore tier and never once execute the spill path — and would
  discover it at 16-14, which is exactly the shape of 17-12's exit-137
  failure. **16-01 Task 5 measures where upstream flips tier, and 16-05 test 4
  asserts which tier each side used**, so a fixture that silently stayed incore
  fails rather than passes.
* **16-06 (UHF) carries three `vvvv`-class tensors** (`Wvvvv`/`WvvVV`/`WVVVV`,
  `kccsd_uhf.py:1110`) → ~3×. **16-07 (GHF) doubles both `nocc` and `nvir`**,
  so every `nvir⁴` tensor is **16×** its RHF counterpart: diamond `gth-szv`
  2×2×2 GHF `vvvv` is 32 MiB, `gth-dzvp` 2×2×2 GHF `vvvv` is **28.6 GiB**. The
  tier branch is not optional in 16-07 under any fixture, and its plan says so.

### 2.4 Upstream's own memory estimate is a documented TODO — do not port it

`kccsd_rhf.py:1100-1107` returns `nkpts³ · nmo⁴ · 4 · 16` bytes and carries
`# TODO: Improve incore estimate and add outcore estimate`. Against the table
above:

| cell / basis | mesh | `_mem_usage` | actual 7 blocks | over-estimate |
|---|---|---|---|---|
| diamond `gth-szv` | 2×2×2 | 128 MiB | 14 MiB | **9.1×** |
| diamond `gth-dzvp` | 2×2×2 | 13.9 GiB | 2.26 GiB | **6.2×** |

Porting it literally imports a 6-9× over-estimate into this port's **HARD**
refusal — i.e. it would refuse jobs that fit, on a machine already short of
memory. **Ruled (D-PBC-29 clause 4):** the tier is chosen from a per-tensor
exact byte count (`16-02` Task 3, `16-CONTEXT §3.6`). 16-01 Task 5 records
upstream's factor per row so the divergence is deliberate and documented rather
than silent.

---

## 3. `symm_map` — a genuine ~4×, absent from `§8.8` entirely

This is the phase's single largest speed item and `§8.8` does not mention it.

`kpts_helper.py:583-612` assigns each of the `nkpts³` k-triples to an orbit of
at most **four**, generated by the four `transform_symm` operations
(`:614-630`): identity, `transpose(2,3,0,1)`, `conj(transpose(1,0,3,2))`,
`conj(transpose(3,2,1,0))`. `kccsd_rhf.py` uses all four: `:783` builds the
map, `:798`/`:804` iterate representatives and their orbits, `:805` and `:909`
apply the transform. So upstream performs `≈ nkpts³/4` integral transforms and
obtains the rest by transposition and conjugation — arithmetic that is free
next to an AO→MO transform.

**`15-REVIEW.md D-15-R-04` ruled this saving ≤2× and that ruling does not carry
over.** Its reason was specific to KMP2: ops 2 and 3 map an `(ov|ov)` block to
a `(vo|vo)` block, which KMP2 never asks for. **KCCSD asks for the full general
block** — `oooo`, `ooov`, `oovv`, `ovov`, `voov`, `vovv`, `vvvv`
(`kccsd_rhf.py:789-794`) — so all four operations land inside the set it wants.
The two phases reach opposite conclusions from the same helper, correctly.

Caveat, stated rather than hidden: the orbit is 4 *generically* and collapses
at fixed points (`kp == kq` with `kr == ks`, and the Γ-containing triples), so
the realised ratio is somewhat under 4 and shrinks as `nkpts → 1`. **16-01
Task 6 measures `len(symm_map)` against `nkpts³` and wall-clocks both paths at
`nkpts` = 2, 4, 8, 27** rather than asserting the factor cold — and if the
measured ratio is materially below 4, it is reported. `15-REVIEW.md D-15-R-04`
is the precedent for a corollary whose arithmetic was backwards even though its
conclusion survived, and it is worth not repeating in the other direction.

**Ruled (D-PBC-29 clause 3):** `symm_map` is used from the first version of
16-05 and 16-06, not retrofitted, with a bit-identity test against the
all-triples path (16-05 test 5).

---

## 4. EOM and (T) — where the memory actually goes

### 4.1 The Davidson subspace is the EOM wall, not the EOM Hamiltonian

The obvious worry — that EOM forms an `(nkpts·nocc·nvir)²` Hamiltonian — is
handled by construction: `eom_kccsd_ghf.py:128` and `:1352` drive
`lib.davidson_nosym1`, which is matrix-free. The real allocation is the
Davidson **subspace**, and it is not small.

Upstream's exact vector sizes:

```
eom_kccsd_rhf.py:412   IP:  nocc + nkpts**2 * nocc**2 * nvir
eom_kccsd_rhf.py:812   EA:  nvir + nkpts**2 * nocc * nvir**2
eom_kccsd_ghf.py:753   IP:  nocc + nkpts*nocc*(nkpts*nocc-1)*nvir//2
eom_kccsd_ghf.py:1270  EA:  nvir + nocc*nkpts*nvir*(nkpts*nvir-1)//2
```

`davidson_nosym1` holds both the expansion space and its images
(`xs` and `ax`), i.e. `2 · max_space · nroots` vectors, with `max_space = 20`
by default (`linalg_helper.py:742`). Derived, complex128:

| cell / basis | mesh | one IP vector | one EA vector | subspace, `nroots = 6` |
|---|---|---|---|---|
| diamond `gth-szv` | 2×2×2 | 66 KiB | 66 KiB | 16 MiB (IP), 16 MiB (EA) |
| diamond `gth-dzvp` | 3×3×3 | 4.1 MiB | **22.6 MiB** | 985 MiB (IP), **5.4 GiB** (EA) |

So `max_space` is a **real memory knob**, not a constant to inline — which is
why 16-03 Task 1 requires it, `lindep`, `lessio` and `follow_state` to be
ported as parameters rather than waved through as defaults. 16-09/10/11 each
carry a peak-memory assertion against this bound.

EE is larger again and its size is *not* a closed form — `EOMEESinglet`
computes it by an explicit `loop_kkk` with a `kika == kjkb` / `kika > kjkb`
packing rule (`eom_kccsd_rhf.py:1434-1476`). **16-01 Task 4 measures it by
calling upstream's own `vector_size`** rather than this review deriving it;
that packing is also exactly where an off-by-one silently changes the vector
length, which is why 16-10 test 3 is a round-trip on the asserted length.

### 4.2 (T) is a streaming problem, and the blocking is the algorithm

`kccsd_t_rhf.py:236` hands `CCsd_zcontract_t3T` 24 raw pointers plus
`mo_offset` and `slices` (`:229-245`). That blocking is not an optimisation
bolted onto a simple loop: `t3` is `nocc³ · nvir³` **per k-triple**, formed,
consumed and discarded. A port that materialises `t3` over all k turns a
streaming problem into an allocation no `§9.2` fixture can hold.

**Ruled:** 16-08 Task 2 ports the blocking, and **16-08 test 6 asserts peak
live bytes stay proportional to one block's `nocc³·nvir³`, not to
`nkpts³·nocc³·nvir³`** — which makes the claim testable rather than
aspirational. 16-08 Task 1 also requires `kccsd_t_rhf_slow.py` to be ported
**first**, because it is the only oracle-free reference for the blocked path
and porting the fast path first is how a blocking bug ships.

---

## 5. Contractions — host rayon, not `zgemm_dense`

Two standing measurements in this workspace decide this, and neither is in
`§8.8`:

* `zgemm-dense-loses-to-host-rayon` — `zgemm_dense` measured **6-12× slower**
  than a host rayon loop on the CPU backend, **and 1.35e-10 off** on grid
  reductions, i.e. outside this project's 1e-11 gate.
* `pyscf-algebra-cpu-is-default-backend` — the CPU runtime *is* the default
  here, so that measurement is the one that applies.

**Ruled (D-PBC-29 clause 2):** every Phase-16 contraction is a host rayon loop
over k-point triples with `oracle_*` accumulators. This is not a permanent
verdict on `zgemm_dense`; 16-14 Task 4.2 **re-measures it against the phase's
actual contraction shapes** and amends the ruling if it wins there.

The companion requirement is `16-CONTEXT §3.2`: **every contraction site names
its primitive.** `oracle_zdot` is `zdotc` (`zoracle.rs:36`) and
`15-REVIEW.md D-15-R-02` found that following a plan that says only "route
through `oracle_dot`" produces `Σ x·x` instead of `Σ conj(x)·y` — a plausible
wrong number that only the final energy gate catches. Phase 16 has many more
such sites than Phase 15 and a genuine mix of both conjugations, so no plan in
this set is allowed to say only "route through `oracle_dot`", and 16-04's
verification greps for it.

---

## 6. RULING — **D-PBC-29**

`D-PBC-28` is taken by Phase 15's speed ruling (`15-CONTEXT §7`, itself
renumbered from 27 by `15-REVIEW.md D-15-R-01` after 17-10 claimed 27). Phase
16's ruling is **D-PBC-29**, four clauses:

1. **Complex tensors get their own arena.** `ZWorkspacePool` in
   `pyscf-runtime` with `shape_bytes = product * 16`, non-copying block
   access and per-buffer locking; `WorkspacePool`'s f64 API unchanged.
   Reinterpreting `Box<[f64]>` as complex pairs is forbidden — it halves the
   number that reaches the HARD refusal (§2.1).
2. **Contractions are host rayon loops with `oracle_*` accumulators, not
   `zgemm_dense`** (§5), and every site names its primitive. Re-measured at
   16-14 Task 4.2.
3. **`symm_map` is used from the first version** of the ERI build — a derived
   ~4× on the phase's dominant step, measured by 16-01 Task 6 (§3).
4. **Storage tiers are selected from an exact per-tensor byte count**, never
   from upstream's documented-TODO `_mem_usage` (§2.4), and **at least one
   green test must cross a tier boundary** (§2.3).

Evidence: `workspace_pool.rs:266-274`/`:278-280`/`:397`/`:461-483`;
`kpts_helper.py:583-630`; `kccsd_rhf.py:132-137`, `:179-192`, `:423-455`,
`:777-832`, `:1100-1107`; `kccsd_t_rhf.py:229-245`;
`eom_kccsd_rhf.py:412`/`:812`; `eom_kccsd_ghf.py:753`/`:1270`;
`linalg_helper.py:742`; the standing memories
`zgemm-dense-loses-to-host-rayon` and `pyscf-algebra-cpu-is-default-backend`.
Gated by: 16-02, 16-05 tests 4-5, 16-08 test 6, 16-09/10/11's memory bounds,
16-14 Task 4.

---

## 7. Findings recorded, deliberately NOT acted on

### 7.1 `kccsd_rhf.py:461`'s commented-out `kconserve_pmatrix`

`#:Ps = kconserve_pmatrix(cc.nkpts, cc.khelper.kconserv)` — a dead vectorised
momentum-projection alternative to the explicit k-loop. It is commented out
upstream, so RULE 2 says it is not part of the port. Recorded because a future
reader will see it and wonder; it is not a hidden speed lever, it is an
abandoned experiment.

### 7.2 The `EOMIP_Ta` / `EOMEA_Ta` variants are cheap and are in scope anyway

`eom_kccsd_ghf.py:760` and `:1277` add a `(T)*(a)` correction on top of the
base EOM classes. They reuse the base matvec entirely, so they cost almost
nothing beyond the class, and 16-09 ships them with the base. No optimisation
decision needed — noted so a later reader does not mistake them for scope
creep.

### 7.3 The `lessio` flag is a real memory/recompute trade and is left to a knob

`davidson_nosym1(.., lessio=False)` trades stored subspace images for
recomputation. Given §4.1's 5.4 GiB EA subspace it is potentially the
difference between running and not running — but the correct default is
upstream's, and choosing otherwise before anything has been measured on real
hardware would be guessing. **16-03 ports it as a parameter with upstream's
default**; whether to flip it is a 16-14 question with a measurement behind it,
not a plan-time decision.

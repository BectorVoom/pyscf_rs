# Phase 15 — plan review + speed optimisation pass

**Written:** 2026-09-02, before any Phase-15 code.
**Scope:** all seven plan files + `15-CONTEXT.md` as they stood on 2026-09-01,
reviewed against the vendored PySCF 2.12.1 tree, the current Rust workspace, and
the standing measurements this repo has committed.
**Outcome:** 6 defects fixed in the plans, 1 new plan added (15-08), 3 findings
recorded but deliberately not acted on.

Every claim below carries the file and line that proves it. Nothing here is a
measurement of new code — no Rust was written or run.

---

## 1. What the plans got right, so it does not get lost

The pre-implementation discipline in `15-CONTEXT.md` is the best thing in the
phase and none of it needed changing:

* §1.1-§1.3's three scope corrections (`kconserv` already shipped, 15-02 largely
  shipped by 14-05, `KUMP2`'s kernel absent upstream) are each verified — the
  files and line numbers check out.
* §2's refusal to accept `1e-14`-vs-`1e-8` without a measurement, and 15-01's
  measure-then-gate ordering, are exactly the Phase-14 lesson applied.
* §3.1 (column-major vs row-major `mo_coeff`), §3.3 (`LARGE_DENOM` is
  arithmetic, not a guard) and §3.4 (virtual-TOP-aligned padding) are the three
  traps most likely to produce a plausible wrong number, and all three are named
  with the upstream line.
* 15-03 Task 5 test 8 (`CᴴSC = I` instead of a round-trip) and 15-05 test 4
  (supercell equivalence) are the two highest-value oracle-free tests in the
  phase.

---

## 2. Defects found, and what was changed

### D-15-R-01 — `D-PBC-27` is already taken. SEVERITY: high (silent doc collision)

`15-CONTEXT §7` opened *"RULING (adopt as D-PBC-27)"* and five plan files
referenced that number. **`D-PBC-27` was allocated on 2026-09-01 to plan
17-10's `RsCell`/`ExtendedMole` ruling** (`PBC-MASTER-PLAN.md` §6, the decision
table's last row; also `ROADMAP.md`'s Phase-17 entry). Two unrelated rulings
under one identifier, in a project whose whole audit story is "you can look the
decision up a year later."

**Fixed:** the Phase-15 speed ruling is now **D-PBC-28** in `15-CONTEXT §7` and
in 15-01/02/04/05/07. §7's opening paragraph records the renumber so a summary
written before today still resolves.

### D-15-R-02 — `oracle_zdot` is `zdotc`; every KMP2 contraction is `zdotu`. SEVERITY: high (wrong number, no test catches it)

`crates/pyscf-algebra/src/zoracle.rs:35` — the workspace's **only**
bit-deterministic complex inner product computes `Σ conj(x)·y`
(`re = dot(xr,yr) + dot(xi,yi)`). `zblas::zdotu_dense` is unconjugated but
`zoracle`'s own module doc (`:1-14`) rules the device reductions out for
anything that lands in an energy. There is no `oracle_zdotu`.

Three of Phase 15's four hot contractions have **no conjugate**:

| contraction | upstream | why unconjugated |
|---|---|---|
| `Lov·Lov → oovv` | `kmp2.py:96` | `df_ao2mo.rs:12-19` — the conjugation is already inside `cderi` |
| `edi = 2·Re⟨t2, oovv[ka]⟩` | `kmp2.py:113` | `t2` was already conjugated at `:110` |
| `exi = −Re⟨t2, oovv[kb]ᵀ⟩` | `kmp2.py:114` | same |

15-05 Task 3 said only *"route the inner contractions through
`pyscf_algebra::oracle_dot`"*. Following that literally with the complex sibling
computes `Σ (oovv/e)·oovv` — the unconjugated square — which runs, returns a
plausible number, and is wrong. The only gate that would notice is the final
`e_corr`, i.e. the last thing measured in the phase.

**Fixed:** new `15-CONTEXT §3.10` states the trap and the two ways out;
**15-04 gains Task 2b**, which adds `oracle_zdotu` / `oracle_zdotu_re` /
`oracle_zdot_re` to `pyscf-algebra` with a bit-identity test
(`oracle_zdotu(conj(x),y) == oracle_zdot(x,y)`) *before* anything consumes them;
15-05 Task 3 now names the primitive per site. The `*_re` variants are a free
2× on `edi`/`exi`, which discard two of the four `oracle_dot` calls.

### D-15-R-03 — the non-DF route materialises the `nao⁴` AO ERI and upstream does not. SEVERITY: high (this is the phase's largest speed item)

`crates/pyscf-pbc-df/src/pbc_ao2mo.rs:449-457`:

```rust
pub fn fft_general(df, mos, kptijkl) -> Result<Eri, _> {
    let ao = fft_get_eri(df, kptijkl)?;   // FULL nao² × nao²
    Ok(mo_eri(&ao, nao, mos))             // then transform_ao_eri
}
```

Upstream's `fft_ao2mo.general` (`fft_ao2mo.py:145-152`) transforms AO→MO **on
the real-space grid first** (`mos = [lib.dot(c.T, aos[i].T) ...]`) and hands the
MO-resolved grid arrays to `_contract_plain`; it never builds the AO block.
`aft_ao2mo.general` is the same shape.

`kmp2.py:98-104` calls this **`nkpts³` times**, and it is the route the phase's
own committed anchor uses (`kmp2.py:820`, FFTDF, `exxdiv=None`). Derived counts
(§7.0 of the context; not measurements):

* flops `Ng·nao⁴` vs `Ng·(nocc·nvir)²` — ratio `(nao²/(nocc·nvir))²`, i.e.
  **16×** on diamond `gth-szv` and **59×** on a `gth-dzvp` cell.
* memory `nao⁴` complex per call — 1.6 GB at `nao = 100`, on a machine where
  17-12's host suite already exit-137s (`STATE.md`) and where the standing note
  `materialised-grid-values-oom` records the identical failure mode.
* it is also an **unrecorded RULE-2 structural deviation** shipped by 14-05.

**Fixed:** new **plan 15-08** (`wave: 0`, `depends_on: []`) ports the MO-first
path, keeps the AO-ERI route reachable, bit-compares the two, and measures the
ratio at two bases. 15-05 now `depends_on` it and says what its test 7b means if
15-08 has not landed. `15-01` gains **Task 7b**, which prices upstream's own two
calls so 15-08 has a target rather than an argument.

### D-15-R-04 — the `symm_map` corollary's arithmetic is backwards. SEVERITY: medium (right conclusion, unusable reasoning)

`15-CONTEXT §7`'s corollary declined the ERI symmetry because *"the dominant
cost per triple is the `nocc²·nvir²` `edi`/`exi` contraction, which every member
of a symmetry orbit still has to pay."*

Counting: the assembly is `naux·(no·nv)²` (DF) or `Ng·(no·nv)²` (non-DF); the
`edi`/`exi` pair is `2·(no·nv)²`. **The assembly is `naux/2` — or `Ng/2` —
times MORE expensive**, not cheaper. The symmetry would shrink the dominant term.

The conclusion survives for a different, true reason: of `transform_symm`'s four
operations, only op 0 and op 1 (`transpose(2,3,0,1)`, `(ia|jb) → (jb|ia)`) map an
`(o v | o v)` block to another `(o v | o v)` block. Ops 2 and 3
conjugate-transpose *within* each pair and yield `(vo|vo)` blocks KMP2 never
asks for. **The usable saving is ≤ 2×, not 8×**, against `build_symm_map`'s own
`O(nkpts³)` cost and a second `nkpts`-long buffer.

**Fixed:** `15-CONTEXT §7.7` states the correct arithmetic and the correct
reason; 15-05 Task 3 requires the *correct* reason in the code comment and logs
the ≤2× as a Phase-16 carry-over (KCCSD pays `build_symm_map` anyway, so the
trade there is genuinely different).

### D-15-R-05 — 15-02's test 4 forces a dependency the same plan promises not to add. SEVERITY: medium (build-time cost + internal contradiction)

15-02 Task 3: *"`pyscf-pbc-lib`'s dependency list does not change."*
15-02 Task 4 test 4: *"take a real `Eri7d` block from
`pyscf_pbc_df::df_ao2mo::ao2mo_7d`"* — inside `crates/pyscf-pbc-lib/tests/`.

`pyscf-pbc-df/Cargo.toml:19` already depends on `pyscf-pbc-lib`. Cargo *permits*
the dev-dependency cycle, so it would compile — but `cargo test -p pyscf-pbc-lib`
would then build the entire DF stack, on this phase's critical path, for one
test. Separately, Task 2's `symm_map: Option<IndexMap<…>>` needs an `indexmap`
dependency the same crate does not have.

**Fixed:** test 4 moves to `crates/pyscf-pbc-df/tests/khelper_symm.rs` (added to
`files_modified`), with the reason written in the test's own description; the
container becomes the `Vec<(key, Vec)> + HashMap` index form the plan already
offered as an alternative, so no `indexmap` is added. The other six tests stay
where they were and need nothing but the crate itself.

### D-15-R-06 — 15-05 and 15-07 mandate an FMA check `AGENTS.md` waives. SEVERITY: low (contradicts a project instruction)

15-05 Task 5 test 13 required adding `pyscf-pbc-mp` to `check-no-fma`'s
`SCAN_TARGETS`; 15-07 Task 4 ran `check-no-fma` in the closing gate.
`AGENTS.md` ends with **"Do not need to check FMA."**

**Fixed:** both removed. The removal is *stated* — in 15-05 test 13's slot and
in a 15-VERIFICATION §1 row — rather than silently dropped, so a reader
comparing against Phases 13/14 (which did run it) sees a ruling, not a gap.

---

## 3. Speed optimisations added beyond the defects

All eight are now `15-CONTEXT §7`, sub-numbered, each landing in a named plan
and reported in 15-07 §8's ledger. Summarised:

| § | ruling | lands in |
|---|---|---|
| 7.1 | parallel **granularity**: `nkpts²` is 4 tasks on the `[1,1,2]` anchor; nest `ka`, and parallelise `r_e2` over its auxiliary index (`df_ao2mo.rs:349-392`, disjoint output slices, one line) | 15-04, 15-05 |
| 7.2 | the reduction obligation needs a primitive that does not exist (D-15-R-02) | 15-04 Task 2b |
| 7.3 | **no `zgemm_dense`** — measured 6.7-8.3× slower than a rayon host loop on the default CPU backend, and 1.35e-10 off the pairwise tree (`.planning/pbc/baselines/contract-mesh{21,31}.json`) | 15-04, 15-05, 15-08 |
| 7.4 | `Lov` stores `L` **fastest** — `(nocc·nvir, naux)`, deviating from `kmp2.py:190` — so §7.3's ordered dot is contiguous | 15-04 Task 3 |
| 7.5 | the MO-first transform (D-15-R-03) | 15-08 |
| 7.6 | hoist: `fft_get_eri` rebuilds `Gv`/`weights`/`coulG` on **every** call (`pbc_ao2mo.rs:246-262`) — `nkpts³` times for at most `nkpts` distinct `q`; and `kmp2.py:99-102` re-slices `orbo`/`orbv` per triple | 15-08 Task 2, 15-05 Task 3 |
| 7.7 | the corrected `symm_map` reasoning (D-15-R-04) | 15-05 Task 3 |
| 7.8 | the rayon fan-out multiplies peak memory: `min(threads, nkpts²)·nkpts·(no·nv)²·16` bytes. Add it to the pre-flight and **bound the pool** — a refusal is correct, an exit-137 is not | 15-05 Task 3 |

### Plan-execution speed (the critical path, not the code)

The dependency graph had six strictly sequential waves for seven plans, and two
of the edges were not real:

* **15-02 (`KptsHelper`) had `depends_on: [15-01]` and reads nothing 15-01
  measures.** It is pure combinatorics over a table plan 09-07 already shipped.
  Moved to `wave: 0`, `depends_on: []` — it now runs beside 15-01, which is a
  long Python-only measurement.
* **15-08 is new and depends on nothing** — also `wave: 0`. The largest speed
  item therefore starts at the earliest possible moment rather than last.
* **15-03's edge on 15-01 is narrow** (tests 4-6 only) and **15-04's is
  narrower still** (Task 5 test 9's tolerance). Both plans now say so in their
  objectives and instruct the executor to land the independent tasks first.

This matters more than usual here: `ROADMAP.md`'s Phase-17 process finding
records that this environment restarted every ~20-40 minutes and killed four
consecutive agents in their reading phase, and that *"every plan that succeeded
landed code incrementally."*

---

## 4. Recorded, deliberately NOT acted on

1. **`Gdf::sr_loop` returns owned `Vec<SrBlock>` per `(ki,kj)` call**
   (`gdf/mod.rs:60`), so `Lov`'s `nkpts²` builds copy `cderi` out of the
   `OnceLock` rather than borrowing it. Probably a real cost at scale. Not
   changed: it touches a shipped, gated Phase-14 API with its own callers, and
   Phase 15 has no measurement saying it matters. **15-04's summary should
   report whether it showed up in the profile**; if it did, it is a Phase-16
   item (KCCSD hits the same path far harder).
2. **`transform_ao_eri`'s four half-transforms** (`pbc_ao2mo.rs:288`) are
   serial. Left alone — 15-08 makes the whole function cold for KMP2, and
   optimising a path you are about to route around is wasted work.
3. **Batching the `ka` GEMMs** in the DF route's first pass into one
   `(nocc·nvir) × naux × nkpts·(nocc·nvir)` product per `(ki,kj)`. Bit-parity
   safe (the reduction axis is unchanged; only the output is stacked) and a real
   cache win. **Not adopted**, because §7.3 rules out the GEMM primitive that
   would make it pay, and with a rayon-over-output-rows loop the batching buys
   only locality. Revisit if `krks_profile contract` ever re-opens the device
   GEMM route.

---

## 5. Gate statements: unchanged, and why

**No tolerance in this phase was touched.** §2.1's `1e-14`-vs-`1e-8` conflict
stays open on purpose — 15-01 exists to close it with a measurement, and
pre-empting it here would be the exact failure the plan was written to avoid.
§2.2's per-DF-route rule is untouched and correct.

The one gate-adjacent addition is **15-08 test 2**, which explicitly says the
MO-first and AO-ERI routes are **not** expected to be bit-identical (they
associate the same sum differently) and that the residual must be measured
before it is gated — the same discipline 13-VERIFICATION and 14-VERIFICATION
both had to apply to mesh ladders.

---

## 6. Files changed by this pass

| file | change |
|---|---|
| `15-CONTEXT.md` | new §3.10 (the `zdotu` trap); §7 rewritten as D-PBC-28 with §7.0-§7.9; §6 cross-reference |
| `15-01-PLAN.md` | D-PBC-28; new Task 7b (price upstream's MO-first `ao2mo`) |
| `15-02-PLAN.md` | `wave: 0`, `depends_on: []`; test 4 relocated to `pyscf-pbc-df`; no `indexmap` |
| `15-03-PLAN.md` | critical-path note (Tasks 1-4 do not wait on 15-01) |
| `15-04-PLAN.md` | new Task 2b (`oracle_zdotu`); `Lov` `L`-fastest layout; `r_e2` parallelism; test 5b; test 9b now reports wall clock; critical-path + 15-08 coordination notes |
| `15-05-PLAN.md` | `depends_on` += 15-08; the primitive named per site; the third memory formula; hoisting; nested parallelism; corrected §7.7 reasoning; FMA test removed |
| `15-07-PLAN.md` | `depends_on` += 15-08; `check-no-fma` removed with a stated reason; §8 ledger expanded to one row per §7.9 item; oracle suite gains the 15-08 case; §5 gains the gamma-shortcut non-port |
| `15-08-PLAN.md` | **new** — the MO-first transform + per-`q` `CoulGCache` |
| `15-REVIEW.md` | this file |

**Not yet propagated:** `PBC-MASTER-PLAN.md` §6 (record D-PBC-28 next to
D-PBC-17/D-PBC-27), §8.7's plan table (five plans → eight), and `ROADMAP.md`'s
Phase-15 sentence. 15-07 Task 3 already owns all three; doing them now would
pre-empt 15-01's gate restatement, which lands in the same edit.

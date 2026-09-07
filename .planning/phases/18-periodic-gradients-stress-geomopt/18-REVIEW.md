# Phase 18 — plan review + speed & memory optimisation pass

**Written:** 2026-09-02, before any Phase-18 code.
**Scope:** `PBC-MASTER-PLAN §8.10`'s seven-plan Phase-18 table, its cintx
posture note and its multigrid note, reviewed against the vendored PySCF 2.12.1
tree, the vendored `cintx` tree and the current Rust workspace; then a
speed-and-memory pass over the fifteen-plan replacement in `18-CONTEXT §4`.
**Outcome:** 8 defects in `§8.10` (all fixed in the new plan set, 3 of them
scope-level), 7 speed/memory rulings folded into the plans as **D-PBC-30**,
4 findings recorded but deliberately not acted on.

Every claim carries the file and line that proves it. **Nothing here is a
measurement of new code** — no Rust was written or run. The numbers below are
*derived* byte counts and call counts, and 18-01 measures the ones that can be
measured before implementation starts.

---

## 1. What `§8.10` got right, so it does not get lost

* **The line-count inventory is exact.** Every figure in the table reproduces
  `wc -l` to the line — `krhf` 418, `rhf` 188, `kuhf` 124, `uhf` 103, `krks`
  141, `kuks` 135, `krks_stress` 404, `kuks_stress` 308, `rks_stress` 462,
  `uks_stress` 246, `geometric_solver` 246. Total `pyscf/pbc/grad/*.py` = 2,848
  and `pyscf/pbc/geomopt/*.py` = 269.
* **"FFTDF J/K gradients are grid-based, not integral-based … `int2e_ip1` is
  never called on the FFTDF path"** is correct and load-bearing, and it is the
  single most useful sentence in the section. It is what collapses this phase's
  two-electron derivative-integral surface to zero. §2.2 below is the
  consequence it did not draw.
* **"Stress needs no derivative integrals at all"** is correct, and the proof
  is exactly where `§8.10` says it is (`rks_stress.py:95-112`, though the
  functions are at `:86-111` in 2.12.1). What it costs instead is a new AO
  family, which `§8.10` does not mention — `18-CONTEXT §1.6`.
* **The standing `verify_fd` rule** — *"`pyscf-pbc-grad` MUST expose the same
  `verify_fd` on every gradient it ships, using `Cell`-aware scanners. Do not
  ship an analytic gradient without it."* — is right, is the correct primary
  gate for this phase, and survives unchanged. Only its *tolerance* was wrong
  (`18-CONTEXT §2`).
* **The multigrid note added by 17-12 is right about which half is inherited**
  (v2, not v1) and right that v2 must not be expected to be fast. §3.7 records
  the one thing it did not carry forward.

---

## 2. Structural defects in `§8.10`

Recorded here in review form; the evidence is in `18-CONTEXT §1`.

| # | defect | consequence | fixed by |
|---|---|---|---|
| D1 | The gamma half is a multigrid-v2 program and **five v2 gradient entry points do not exist** (`pbc/grad/rhf.py:42-47` asserts `isinstance(ni, MultiGridNumInt2)`, `else: raise NotImplementedError`) | ~209 upstream lines costed at zero; `grad/rhf`+`grad/uhf` are unstartable | new plan **18-09** |
| D2 | `get_jk_e1`/`get_j_e1`/`get_k_e1` exist **only on FFTDF** (`fft.py:324-340`); GDF/RSGDF/MDF/AFTDF are not subclasses | a `.density_fit()` user gets `AttributeError`, and nothing says so | **18-04** Task 4, a named refusal |
| D3 | The two cintx blockers are **already resolved**, at the same maturity as families this port ships today (`api_manifest.csv`, `pyscf-pbc-gto/Cargo.toml:70`) | the phase was planned around a `#[ignore]` that is not needed | **18-03**; retires `§2.4`'s two "BLOCKS plan 18-01" rows |
| D4 | Half of 18-01 is **already shipped** — `int1e_ipovlp`/`int1e_ipkin`/`int1e_ipnuc` are in `SUPPORTED_INTORS` (`pbc_intor.rs:275-277`) with a test (`tests/pbc_intor.rs:375`) | over-costed | folded into **18-03** |
| D5 | **`rks_stress` is the base of the other three** and is scheduled third (`krks_stress.py:74-83` and two mirrors import eight symbols from it) | the same ordering defect as `16-CONTEXT §1.4` | **18-12** before **18-13** |
| D6 | The **strain-tensor AO family is a C kernel with no Rust counterpart** (`pyscf/lib/pbc/grid_ao.c:431`; `grep -rn strain_tensor crates/` is empty) | the whole stress half costed at zero | new plan **18-11**, in `pyscf-kernels` per ALG-06 |
| D7 | 18-06's **"add lattice degrees of freedom"** is not in upstream — `grep -rn lattice pyscf/pbc/geomopt/*.py` hits once, inside `__main__` | invented scope with no oracle, forbidden by D-PBC-15 | **18-14** ports the atom-coordinate optimizer only |
| D8 | The **gate is 1e-14 Ha/Bohr** (`ROADMAP:464`) against `FD_TOL = 1e-6` (`verify_fd.rs:35`), `§8.10`'s own 1e-6, and upstream's 5e-7 — and is below the arithmetic floor of a central difference | rejects correct and incorrect implementations alike | **18-01** measures; `18-CONTEXT §2.3`'s five gates replace it |

---

## 3. Speed and memory — seven rulings (D-PBC-30)

### 3.1 Where the memory actually goes

Derived footprints. Complex arrays at 16 B/element, real at 8. `blk` is the
grid block size, `ngrids` the full uniform grid.

**The stress `get_vxc` block, k-point version** (`krks_stress.py:169-171`) holds
two arrays at once:

| array | shape | GGA element count |
|---|---|---|
| `ao_ks` from `block_loop(..., deriv+1)` | `(nkpts, comp₂, blk, nao)` | `nkpts · 10 · blk · nao` |
| `ao_ks_strain` from `_eval_ao_strain_derivatives` | `(nkpts, 3, 3, comp₁, blk, nao)` | `nkpts · 36 · blk · nao` |

diamond `gth-dzvp`, `nao = 26`, 2×2×2 (`nkpts = 8`), `blk = 8000`:
`8 · 46 · 8000 · 26 · 16` = **1.14 GiB per block**, of which 914 MiB is the
strain array.

**`get_k_e1_kpts`** (`fft_jk.py:346-350`, `:391`):

| array | shape | diamond `gth-dzvp`, `ngrids = 24389` |
|---|---|---|
| `ao1_kpts` (all k, `deriv=1`) | `(nkpts, 4, ngrids, nao)` | 310 MiB @ 2×2×2 · **1.02 GiB @ 3×3×3** · 2.43 GiB @ 4×4×4 |
| inner `rho1`, `blksize = nao` | `(3, blk, naoj, ngrids)` | **755 MiB** per k-pair (`naoj = nao`) → **116 MiB** (`naoj = nocc`) |
| `vR_dm` | `(3, nset, nao, ngrids)` | 30 MiB |

**Stress `rho1`** (`rks_stress.py:191`, `krks_stress.py:161`),
`(3, 3, nvar, ngrids)` real: 6.7 MiB at `ngrids = 24389`, 59 MiB at `60³`.

> **Corrected by Part II §8.1:** the `get_k_e1_kpts` inner peak is ~2× the
> `rho1` figure below (`rho1`+`vG` coexist across the forward transform,
> `vG`+`vR` across the inverse), so 755 MiB is really ~1.5 GiB. The
> `ao1_kpts` figure stands; `ao2_kpts` is a **view** into it, not a second
> table (§8.2).

**The conclusion that shapes every ruling below:** the wall is the *AO tables*,
not the density arrays. `rho1` is a 4–5× saving on a ~60 MiB array; the strain
AO block and `ao1_kpts` are 1-GiB-class. On the machine this project actually
runs on — where 17-12's entire host suite is SIGKILLed at exit 137
(`STATE.md`) — that ordering decides which rulings are worth their complexity.

### 3.2 Clause 1 — the strain block size is chosen by a budget that does not count the strain array

`ni.block_loop` picks `blk` from `max_memory` against the array *it* returns —
`nkpts · 10 · blk · nao` for GGA. The caller then allocates
`nkpts · 36 · blk · nao` beside it (`krks_stress.py:170`) and `block_loop`
never sees it. Upstream's blocking therefore under-counts the block's true
footprint by **4.6× for GGA/MGGA** and **3.25× for LDA** (`4 + 9` vs `4`).

This is not a defect *in* `block_loop` — for its shipped callers there is no
second array. It is a defect in porting the caller as written.

**Ruled (D-PBC-30 clause 1; 18-12 Task 2, 18-13 Task 1):** the stress block
loop sizes its own blocks against the **sum** of both arrays, from
`PYSCF_MAX_MEMORY` (`aftdf.rs:84-85`), not from `block_loop`'s figure. 18-12's
test pins a low budget and asserts the chosen `blk` shrinks accordingly — a
budget that silently ignores the larger array fails a test rather than an OOM
killer.

### 3.3 Clause 2 — build the lattice-image list once for all eighteen strain cells

> **Superseded by Part II §6.2 (D-PBC-31 clause 2)** for `get_ovlp`/`get_kin`:
> upstream's own test carries a *closed form* for these two, asserted at 1e-9,
> so the port does not finite-difference them at all — 36 lattice sums become
> 2 and the error terms below become zero rather than smaller. This clause
> survives for the finite differences that remain, inside the gate. Its
> "1e-8-order discontinuity" is an **upper bound** (`cell.precision`), not a
> realised win — see §6.2's closing paragraph.

`rks_stress.get_ovlp` and `get_kin` (`:86-111`) each run the 9 `(x,y)` strain
components, and each component builds two displaced cells and calls
`pbc_intor` on both — **18 lattice sums per quantity, 36 for the pair**. Every
one of those re-derives `Ls` through `lattice_images` → `get_lattice_ls`,
which this port's own doc comment calls *"an `O(nimgs · natm)` filter"* and
which plan 10-06 already built a reuse hook for:
`intor_cross_with_images(intor, cell1, cell2, kpts, opts, ls, neighbor_list)`
(`pbc_intor.rs:295`), written because *"rebuilding `Ls` … per operator is pure
waste."*

**It is also more accurate, not just faster.** The displacement is
`disp = 1e-5` on lattice vectors of 3–5 Bohr, so `rcut` moves by ~1e-5
relative. An image sitting on the screening boundary can enter one displaced
cell's `Ls` and not the other's, and the finite difference then contains a
step discontinuity of order `cell.precision` (1e-8) — the same order as
upstream's own acceptance for these quantities (`test_rks_stress.py:388`,
`< 1e-8`). A single `Ls` shared by both sides of every difference removes that
term exactly.

**Ruled (D-PBC-30 clause 2; 18-12 Task 1):** `Ls` is built once from the
undisplaced cell and passed to all 36 evaluations through the existing
`intor_cross_with_images`. 18-12's test compares the shared-`Ls` result against
per-cell `Ls` at upstream's 1e-8 and records the difference, so the claim that
this is a no-op-or-better is measured, not asserted.

### 3.4 Clause 3 — `ao1_kpts` is allocated outside the block guard, and upstream's guard is miscomputed anyway

> **Cost model corrected by Part II §7.2 (D-PBC-31 clause 8):** `get_k_e1_kpts`
> is a *double* loop, so the streaming branch costs `nkpts²` AO evaluations,
> not `nkpts`. The two branches become one tiled residency `m`. And per
> §8.1, upstream's `/4` is deliberate buffer multiplicity — only the doubled
> `mem_now` is the defect.

Two separate problems in five lines of `fft_jk.py`:

* **`ao1_kpts` is not blocked.** It is built for **all** k-points at
  `deriv = 1` before the loop (`:346-350`) and is `nkpts · 4 · ngrids · nao`
  complex — 1.02 GiB at diamond `gth-dzvp` 3×3×3. `blksize` (`:363`) guards
  only the inner `rho1`. The grid cannot be blocked here (the FFT at `:392`
  needs the full grid), so the only lever is *how many k-points are resident*.
* **`blksize` subtracts `mem_now` twice.** `:362` is
  `max_memory = mydf.max_memory - mem_now`; `:362` is
  `(max_memory-mem_now)*1e6/16/4/3/ngrids/nao`. Same shape as `16-REVIEW §2.4`
  ("upstream's own memory estimate is a documented TODO — do not port it").

**Ruled (D-PBC-30 clause 3; 18-04 Tasks 2 and 3):** the k-point AO table is a
**budgeted cache** under `PYSCF_MAX_MEMORY` with two branches — resident (all
k-points, when the table fits) and streaming (only the two k-points in flight,
recomputing per pair, peak `2 · 4 · ngrids · nao`, an `nkpts/2` reduction paid
for in `nkpts`× AO re-evaluation). The `blksize` formula is re-derived, not
transcribed. **Both branches must be exercised by a test**: 18-04 pins a low
budget on a reference fixture so the streaming path runs in CI. This is 16-01's
rule — *"a fixture that silently stayed incore fails rather than passes"* —
and it is the specific failure mode 17-12 hit.

### 3.5 Clause 4 — the MO factorisation is worth more here than in the energy; the k-pair symmetry is worth nothing

**Take the MO factorisation.** `fft_jk.py:357-359` collapses the ket index from
`nao` to the occupied count when the density matrix carries `mo_coeff`/`mo_occ`
tags. That turns the inner `rho1` from `3 · nao · nao · ngrids` into
`3 · nao · nocc · ngrids`: **2×** on diamond `gth-szv` (`nao 8 → nocc 4`) and
**6.5×** on `gth-dzvp` (`26 → 4`), i.e. 755 MiB → 116 MiB in the table above.
The precedent exists — 17-10 Task 4 shipped the MO-factorised energy-path
`get_k_kpts` — but the transport does not: this port's `KMats` carries no
`mo_coeff` tag, and upstream's `getattr(dm_kpts, 'mo_coeff', None)` (`:322`) is
a Python attribute on an ndarray with no Rust analogue.

**Ruled (clause 4a; 18-02 Task 3, consumed by 18-04):** 18-02 adds the tagged
density type (upstream's `_tag_rdm1` contract) and 18-04's factorised branch is
gated by a test asserting the tagged and untagged routes agree — and that the
tagged one allocates less.

**Leave the k-pair symmetry alone.** `fft_jk.rs:265-330` documents the
energy-path identity `rho1^{21}[(i,j),g] = conj(rho1^{12}[(j,i),g])`, which
halves the transforms. It holds because bra and ket carry the same AO table. In
`get_k_e1_kpts` the bra is `ao1T[1:,p0:p1]` — the derivative AO — and the ket
is `ao2T`; the `(j,i)` swap moves the derivative to the ket and the identity
does not close. `get_k_kpts_opts`'s own doc block already warns *"This CHANGES
THE RESULT … A gate run with this on must be re-baselined"*, so a wrongly
enabled flag here would move the last bits of a gradient that is otherwise
correct — the hardest class of error to attribute.

**Ruled (clause 4b; 18-04 Task 5):** the gradient route **refuses**
`kk_symmetry` rather than ignoring it, and a test asserts the refusal. A
comment asking implementers not to enable it is not a mechanism.

### 3.6 Clause 5 — fuse the strain AO kernel with the ordinary AO kernel, if 18-01 says it pays

`krks_stress.py:170-173` evaluates the AO table twice over the same block: once
through `block_loop(..., deriv+1)` and once through
`_eval_ao_strain_derivatives(..., deriv=deriv)`. The two share every radial and
exponential factor and differ only in the angular/coordinate weighting. The
port must write the strain kernel from scratch (there is no C counterpart —
`18-CONTEXT §1.6`), so it is free to emit both in one pass.

**This is the one clause held back for measurement.** This project has already
measured that in its collocation kernels *"buffer traffic, not exp, was the
wall-clock sink"* (`materialised-grid-values-oom`). If that holds here, fusing
saves one pass over a 46-component buffer — a real but bounded win — and not
the exponentials, which is where the intuition would have put it.

**Ruled (clause 5; 18-01 Task 6, 18-11 Task 3):** 18-01 measures the split
between exponential evaluation and buffer traffic for a 46-component strain +
AO block. 18-11 ships the fused kernel **only if** the measurement supports it,
and records the number either way. No fusion on intuition.

### 3.7 Clause 6 — hoist the AO table out of `hcore_deriv`'s atom loop

> **Subsumed by Part II §7.1 (D-PBC-31 clause 7):** the matrix this closure
> builds is never needed — `grad_elec` reduces it to three numbers against the
> density, so the whole per-atom loop is one G-space contraction and there is
> no per-atom AO cost left to hoist.

`krhf.hcore_generator` (`:117-147`) returns a closure whose body contains

```python
for kn, kpt in enumerate(kpts):
    ao = eval_ao_kpts(cell, coords, kpt)[0]
```

and `grad_elec` calls that closure once per atom (`:60-68`). The AO table does
not depend on `atm_id`. Ported literally, the cost is **`natm · nkpts`
full-grid AO evaluations plus `natm · 3` inverse FFTs** where `nkpts` and `3`
would do. On the reference cells (`natm = 2`) that is 2×; on any cell worth
optimising a geometry for it is `natm`×.

Hoisting costs memory: the resident table is `nkpts · ngrids · nao` complex —
77 MiB at diamond `gth-dzvp` 2×2×2, 260 MiB at 3×3×3. That is **clause 3's
array again**, at `deriv = 0` instead of `deriv = 1`.

**Ruled (D-PBC-30 clause 6; 18-05 Task 3):** the hoist happens, and it is
governed by clause 3's budgeted cache rather than by a second, independent
memory decision. One cache, one budget, two consumers (`hcore_generator` and
`get_k_e1_kpts`). Stating it once is the point: two budgets that each believe
they own the machine is how a 1-GiB array becomes a 2-GiB one.

### 3.8 Clause 7 — fuse the stress XC contraction into the block loop, and keep the reduction deterministic

`rks_stress.py:191` allocates `rho1 = np.empty((3,3,nvar,ngrids))` over the
full grid, and the only full-grid consumers of it are
`np.einsum('xyng,ng->xy', rho1, vxc)` (`:259`) and, for the `with_j`/`with_nuc`
terms, `rho1[:,:,0]` alone (`:267`, `:276`, `:287`). The XC functional is
pointwise in `rho0`, so `vxc[:,g]` depends only on `rho0[:,g]`: the first
contraction can be evaluated per block and accumulated into the 3×3 output,
leaving only `rho1[:,:,0]` — `9 · ngrids` instead of `9 · nvar · ngrids` —
resident. A **4× (GGA) / 5× (MGGA)** cut on that array.

Two constraints that make this a ruling rather than a suggestion:

* `rho0` must stay full-grid regardless: `pbctools.fft(rho0[0], mesh)`
  (`:266`) needs it, and it is the same size a normal DFT run already carries.
* **Block accumulation changes the summation order** relative to the full
  einsum. The 3×3 accumulator must therefore route through a materialised
  per-block partial array and `pyscf_algebra::oracle_sum` — the D-PBC-17
  pattern this repo applied to `ztrace_ab`/`trace_dm_v` (commit `0bcff45`) and
  which `17-CONTEXT` required of `symmetrize_density` *"from the first version,
  not as a retrofit."*

**Ruled (D-PBC-30 clause 7; 18-12 Task 3):** fuse, keep `rho0` and
`rho1[:,:,0]` full-grid, accumulate through `oracle_sum`. 18-12's determinism
test is the standing one — bit-identical at `RAYON_NUM_THREADS=1` and `8` under
`release-oracle`.

Honest sizing: this is a 4–5× cut on a 6.7 MiB (`ngrids = 24389`) to 59 MiB
(`60³`) array, an order of magnitude below clauses 1 and 3. It is ruled because
it is nearly free once the block loop is being rewritten for clause 1 anyway,
not because it is where the memory is.

---

## 4. Findings recorded, deliberately not acted on

1. **`_contract_vhf_dm`'s screening is a defaulted global.**
   `pbc/grad/rhf.py:30`, `SCREEN_VHF_DM_CONTRA = getattr(__config__,
   'pbc_rhf_grad_screen_vhf_dm_contract', True)`. Screened and unscreened
   contractions differ in the last bits. The port has
   `build_neighbor_list_for_shlpairs` (`neighborlist.rs:221`) so both branches
   are cheap. **Not ruled** because which one is faster depends on the cell's
   sparsity, which nothing has measured on the reference systems — 18-01 Task 4
   measures it and 18-10 picks a default *and states it*. What is not
   acceptable is inheriting a default by transcription.

2. **Upstream's two stress finite-difference steps are not unified, and should
   not be.** `get_ovlp`/`get_kin` hard-code `disp = 1e-5` (`:88`, `:101`); the
   end-to-end tests use `1e-3` (`test_rks_stress.py:403`). Different noise
   floors — an integral difference versus an SCF difference. Unifying them
   would look tidy and would move a gate.

3. **The strain AO array may be algebraically redundant.** **[CLOSED by
   Part II §9 — it is not. 18-11 is required, and there is now a proof rather
   than weak evidence.]**
   `rks_stress.py:215-226` *adds* `einsum('xig,yg->xyig', ao[1:4], coordsT)` to
   `ao_strain`, which is the grid-response term — evidence that the 9-component
   array is built from ordinary derivative AOs and coordinates. If the whole
   `ao_strain` reduced to `ao_deriv(n+1) ⊗ coords`, 18-11 would not be needed
   at all. **Not pursued**: upstream ships a dedicated C kernel
   (`grid_ao.c:431`) rather than that contraction, which is weak evidence the
   identity does not close (lattice-image phase factors are the obvious place
   it would fail). Recorded so that a later session can test it cheaply against
   18-11's own output instead of rediscovering the question.

4. **No HF stress, no periodic Hessian.** `pyscf/pbc/grad/` has no
   `rhf_stress`/`khf_stress`, and `PBC-MASTER-PLAN §8.13` records that
   `pyscf/pbc` has no Hessian module at all — so the missing cintx Hessian
   families (`int1e_iprinvip`, `int2e_ipvip1ipvip2`, …) block nothing in this
   phase. Both are non-ports, not gaps.

---

## 5. Summary of D-PBC-30

| clause | ruling | plan | size of the win |
|---|---|---|---|
| 1 | stress blocks sized against **both** AO arrays, from `PYSCF_MAX_MEMORY` | 18-12, 18-13 | correctness of a 1.14 GiB/block budget that under-counts 4.6× |
| 2 | one `Ls` for all 36 strain lattice sums, via `intor_cross_with_images` **[superseded by D-PBC-31 clause 2 for `get_ovlp`/`get_kin`; survives for the FD inside the gate]** | 18-12 | 36 → 1 `get_lattice_ls`; removes an FD discontinuity bounded by `cell.precision` (an upper bound, not a realised win — §6.2) |
| 3 | k-point AO table is a budgeted cache; both branches CI-exercised; re-derive `blksize` | 18-04 | 1.02 GiB → 76 MiB peak at 3×3×3, paid in recompute |
| 4a | tagged `mo_coeff`/`mo_occ` density; factorised `get_k_e1_kpts` | 18-02, 18-04 | inner `rho1` 755 → 116 MiB (`gth-dzvp`) |
| 4b | gradient route **refuses** `kk_symmetry`, with a test | 18-04 | prevents a silent last-bit corruption |
| 5 | fuse strain + ordinary AO into one kernel pass **iff 18-01 measures a win** | 18-01, 18-11 | one pass over a 46-component buffer; unmeasured |
| 6 | hoist the AO table out of `hcore_deriv`'s atom loop, under clause 3's cache | 18-05 | `natm`× fewer full-grid AO evaluations |
| 7 | fuse the stress XC contraction into the block loop; `oracle_sum` the accumulator | 18-12 | 4–5× on a 6.7–59 MiB array |

---

# Part II — second optimisation pass (D-PBC-31)

**Written:** 2026-09-07, still before any Phase-18 code.
**Scope:** a second speed / memory / **precision** pass over the fifteen plans,
after Part I's D-PBC-30. Part I asked "where does the memory go"; this pass
asks the two questions it did not: **where is the port about to compute a
quantity by finite difference that has a closed form**, and **where is a gate
looser than the upstream test it cites**.

**Outcome:** 12 rulings (**D-PBC-31 clauses 1–12**), one open question from
Part I closed, and three corrections to Part I's own numbers. Every claim
carries the file and line that proves it. **Nothing here is a measurement of
new code.**

The three axes are not independent here and the pass does not pretend they
are: clause 2 is 18× faster *because* it stops finite-differencing, and it is
exact *for the same reason*.

---

## 6. Precision

### 6.1 Clause 1 — Gate A is three tiers upstream, and the plan set flattened it to the loosest

`18-CONTEXT §2.3` states Gate A as **"Target 1e-8, upstream's own number
(`test_rks_stress.py:388`)"**, and 18-11, 18-12 and 18-13 all repeat `< 1e-8`.
Line 388 is real, but it is **one assertion out of twenty-three**, and it is
the loosest one in the file. The actual distribution:

| tier | tolerance | assertions | what they cover |
|---|---|---|---|
| **A1** | **`1e-9`** | `:43 :49 :65 :71` (ovlp), `:79 :85 :101 :107` (kin), `:114 :116` (weight), `:121 :123` (coulG), `:140 :158 :176 :194` (strain AO, sph+cart, deriv 0+1), `:214 :240` (grid response), `:253` (lattice-vector derivatives), `:277 :296 :315` (get_vxc LDA/GGA/MGGA) | every component of the strain machinery |
| **A2** | **`2e-9`** | `:340` (get_j), `:363` (get_nuc) | the two assembled Coulomb terms |
| **A3** | **`1e-8`** | `:388` (get_pp) | the pseudopotential term alone |

Twenty-one of the twenty-three assertions are at `1e-9` or `2e-9`. Gating all
fourteen component tests at `1e-8` is **10× looser than upstream on thirteen of
them**, and it costs nothing to be right: these are the tests that localise a
wrong strain term to a single function, and a 10× margin is exactly where a
wrong term hides.

**Ruled (clause 1; 18-CONTEXT §2.3, 18-11, 18-12 Task 6, 18-13 Task 5,
18-15):** Gate A is **A1 = 1e-9 / A2 = 2e-9 / A3 = 1e-8**, each tier naming the
upstream assertion it inherits. 18-01 measures the port's floor per tier before
any of them is written, exactly as for Gates B–E; if a measured floor sits
above its tier, that is a **finding with a number**, not a licence to fall back
to 1e-8.

### 6.2 Clause 2 — `get_ovlp` and `get_kin` have a closed form, and upstream proves it at 1e-9 in its own test

This is the largest single item in the pass, on all three axes at once.

`rks_stress.get_ovlp` (`:86-97`) and `get_kin` (`:99-111`) compute the strain
derivative of a one-electron integral by **central difference over eighteen
rebuilt cells** — 9 strain components × 2 displacements — and `get_kin` repeats
the same eighteen cell builds. Part I's clause 2 shared the `Ls` across them.

**Upstream's own test does something much better, and asserts it at `1e-9`.**
`test_rks_stress.py:53-71` (and `:88-107` for kin) replaces the whole finite
difference with a closed form and checks it against the FD twice, in two
algebraically distinct arrangements:

```python
bas_coords = np.repeat(cell.atom_coords(), ao_repeats, axis=0)
ovlp10 = np.einsum('xij,iy->xyij', cell.pbc_intor('int1e_ipovlp'), bas_coords)
sc_ovlp01 = intor_cross('int1e_ipovlp', scell, cell)     # scell = supercell over Ls
ovlp01 = np.einsum('xinj,njy->xyij', sc_ovlp01, bas_coords + Ls[:,None])
dat = -(ovlp10 + ovlp01)[0, 1]
assert abs(dat - ref).max() < 1e-9                       # ref = the 1e-5 central difference
```

Derived, the identity is
`dS_k/dε_xy = −Σ_L e^{ik·L} [ ∇_x S[i,j;L]·R_{i,y} + ∇^{ket}_x S[i,j;L]·(R_j+L)_y ]`,
and the weight splits into an **AO-indexed** part `R_{j,y}` and an
**image-scalar** part `L_y`. So the port needs **no supercell and no new
integral family**: `L_y` folds into the Bloch-phase array
`intor_cross_with_images` already builds (`pbc_intor.rs:367-374`), i.e. the
whole 3×3 derivative is **one `int1e_ipovlp` lattice sum evaluated with four
phase rows per k-point** instead of one.

| | upstream, as written | this ruling |
|---|---|---|
| lattice sums, ovlp + kin | **36** | **2** (one per operator, 4 phase rows each) |
| displaced cell builds | 36 | **0** |
| `get_lattice_ls` calls | 36 | 1 |
| truncation error | `O(disp²)` | **none** |
| cancellation error | `2·ε·|S|/2e-5 ≈ 2e-11` | **none** |
| screening discontinuity | ≤ `cell.precision` | **none** |

**Ruled (clause 2; 18-12 Task 1, 18-13 Task 2, and a small hook in 18-02):**
the analytic form is the **production** path for `get_ovlp`/`get_kin`, gamma
and k-point. The finite-difference form is kept as the **test oracle** and
nothing else. The hook is an optional per-image real weight on the Bloch phase
in `intor_cross_with_images` — default `None` reproduces today's behaviour
bit-for-bit, which the existing `tests/pbc_intor.rs` proves on every run.

**The gate is upstream's own number for exactly this substitution: `1e-9`**
(`test_rks_stress.py:65, :71, :101, :107`), and both of upstream's two
arrangements are ported, because a sign error in the ket-derivative term
cancels in one of them and not the other.

**This supersedes Part I's clause 2** for `get_ovlp`/`get_kin`: there is no
longer a 36-way lattice sum to share an `Ls` across. Clause 2 of D-PBC-30
survives only for the finite differences that remain — the ones inside the
gate — where a shared `Ls` still removes a step discontinuity between the two
sides of a difference.

**Correction to Part I while we are here.** `§3.3` claimed the shared `Ls`
"removes a 1e-8-order FD discontinuity". `cell.precision` is the *bound*, not
the realised value: upstream's own analytic-vs-FD assertion passes at `1e-9` on
its fixture, so on that cell the discontinuity is below `1e-9`. Stated as an
upper bound it is honest; stated as a win it was not.

### 6.3 Clause 3 — the k-points move with the strain, and a port that holds them fixed differentiates something else

`krks_stress.get_ovlp` (`:84-98`) does **not** pass `kpts` to the displaced
cells. It passes *re-derived* ones:

```python
scaled_kpts = kpts.dot(cell.lattice_vectors().T)          # :88  fractional
kpts1 = scaled_kpts.dot(cell1.reciprocal_vectors(norm_to=1))   # :93
```

The k-points are held at fixed **fractional** coordinates, so `k·L` is
strain-invariant and the Bloch phase does not participate in the derivative.
A port that strains the cell and reuses the original **Cartesian** k-points
computes a different quantity — one that is correct at Γ (where `k = 0` either
way) and wrong at every other k-point. That is the same failure shape as
18-13 Task 3's `get_ovlp` import and 18-06 Task 1's spin-summing error:
**invisible on the obvious fixture**.

It is also the reason clause 2's closed form is allowed to leave the phase
factor alone — the two facts are the same fact.

**Ruled (clause 3; 18-02 Task 5, 18-13 Task 2):** `_finite_diff_cells`'
contract includes the fractional-k transformation, and 18-02's test asserts
that the two displaced cells' Cartesian k-points differ from the original while
their fractional ones do not. 18-13's k-point stress gate runs at `nkpts > 1`
on a **non-cubic** cell (upstream's own `np.random.seed(5)` lattice), because a
cubic cell makes the discrepancy small rather than absent.

### 6.4 Clause 4 — take the real part inside the contraction, not after it

`krhf.grad_elec:63-66` is three complex contractions per atom, each immediately
reduced by `.real`:

```python
de[x] += np.einsum('xkij,kji->x', h1ao, dm0).real
de[x] += np.einsum('xkij,kji->x', vhf[:,:,p0:p1], dm0[:,:,p0:p1]).real * 2
de[x] -= np.einsum('kxij,kji->x', s1[:,:,p0:p1], dme0[:,:,p0:p1]).real * 2
```

`Re(Σ a·b) = Σ (Re a·Re b − Im a·Im b)` **exactly**, for a fixed summation
order — complex addition is componentwise, so no rounding distinguishes the two.
Computing only the real part halves the multiplies (2 instead of 4 per
product), removes the imaginary accumulator entirely, and leaves `oracle_sum`
running over a real buffer instead of a complex one.

**Ruled (clause 4; 18-05 Task 4, inherited by 18-06/18-07/18-08/18-10):** the
`.real` moves inside the contraction, the D-PBC-17 partial buffer is real, and
a test asserts the result is bit-identical to the complex-then-`.real` route
on a deliberately non-Hermitian density.

### 6.5 Clause 5 — the finite differences divide by the nominal step, and the honest size of that

`verify_fd.rs:105-108` forms `plus[ia][c] += disp` and divides by `2.0*disp`.
The *realised* step is `(x+h) − (x−h)`, which differs from `2h` by up to one
ulp of `x`: at `x ≈ 2` Bohr and `disp = 1e-4`, a relative error of
`4.4e-16 / 2e-4 ≈ 2.2e-12`. On a gradient of order 1 Ha/Bohr that is
**~2e-12 Ha/Bohr**. `_finite_diff_cells`' realised strain has the same shape at
`1e-5`: ~`1e-11` relative.

Both are **two to three orders below** the ~4e-10 cancellation floor 18-01
Task 2 will measure, and below Gate A1's 1e-9.

**Ruled (clause 5; 18-02 Tasks 4 and 5):** use the realised step as the
denominator, because it is free and it removes a term. **Record it as a
sub-floor effect** — 18-01 must not report it as the floor, and no gate is
tightened on the strength of it. It is listed here so that the next session
does not rediscover it and over-claim it.

### 6.6 Clause 6 — `coulG_1` is exactly symmetric and `weight_1` is exactly diagonal

`_get_coulG_strain_derivatives` (`rks_stress.py:112-119`) builds
`coulG_1 = einsum('gx,gy->xyg', Gv, Gv) * coulG_0 * 2/G2` — **symmetric in
`(x,y)` by construction**, 9 components stored where 6 are independent.
`_get_weight_strain_derivatives` (`:121-125`) returns
`weight_1 = np.eye(3) * weight_0` — **exactly diagonal**, and the term it
multiplies (`:260`, `einsum('g,g->', rho0[0], exc) * weight_1`) is therefore a
scalar on the diagonal.

**Ruled (clause 6; 18-12 Tasks 1 and 3):** store and contract the 6 unique
`coulG_1` components and mirror; add the `exc·weight_1` term as a scalar to the
diagonal. `1.5×` on a 1.7 MiB (`ngrids = 24389`) to 15.5 MiB (`60³`) array —
small, but it makes two exact structural facts *explicit* instead of
emergent-to-roundoff, which is what makes an asymmetry in the **total** stress
(which is not symmetric in general) attributable when one shows up.

---

## 7. Speed

### 7.1 Clause 7 — `hcore_deriv` never needs the matrix it builds

Part I's clause 6 hoisted the AO table out of `hcore_generator`'s atom loop.
That is the right direction and it does not go far enough: **there is no reason
to build the matrix at all.**

`krhf.hcore_generator` (`:132-147`) allocates `(3, nkpts, nao, nao)` per atom
and fills it with

```python
vloc = np.einsum('gi,gj,g->ij', ao.conj(), ao, vloc_R, optimize=True)
```

at cost `natm · 3 · nkpts · ngrids · nao²`. `grad_elec:63` then reduces the
whole thing to three numbers against `dm0`. But

```
Σ_ij vloc[k,i,j] · dm0[k,j,i]  =  Σ_g vloc_R[g] · ρ_k[g]
                               =  Σ_G conj(vloc_g[G]) · ρ_k[G]      (Parseval)
```

and **`ρ` does not depend on `atm_id` or on the Cartesian component.** The
entire per-atom loop is one `(natm,3) ← (natm·3, ngrids) × (ngrids)`
contraction against a density the SCF already knows how to build.

| | ported literally | clause 6 (hoist) | this ruling |
|---|---|---|---|
| AO-table evaluations | `natm · nkpts` | `nkpts` | `nkpts` |
| `ngrids · nao²` work | `natm · 3 · nkpts` | `natm · 3 · nkpts` | **`nkpts`** |
| inverse FFTs | `natm · 3` | `natm · 3` | **0** (contract in G-space) |
| resident per-atom array | `3·nkpts·nao²` complex | same | **none** |

On the reference cells (`natm = 2`) the `ngrids·nao²` term falls 6×; on any
cell worth optimising a geometry for it falls `3·natm`×. **This subsumes
clause 6**: there is no per-atom AO cost left to hoist, so clause 6's budgeted
cache is needed for `get_hcore`'s `h1` build and for 18-04, but no longer for a
per-atom hoist.

Two constraints that make this a ruling rather than a rewrite:

* **`hcore_generator` stays a public seam returning the matrix.**
  `grad_elec:68` passes `locals()` to `extra_force`, and 18-08's DFT+U term
  reads it; the Gate-A-style component test needs the matrix too. The fused
  contraction is what `grad_elec` *uses*; the matrix is what the API *exposes*,
  and a test asserts the two agree.
* **G-space and real-space are different summations.** Parseval is exact in
  exact arithmetic and ~1e-13 relative in this one. 18-01 measures the residual
  between the fused and the literal routes; 18-05 gates against the
  measurement. Not bit-identity, and the plan must not claim it.

### 7.2 Clause 8 — the streaming AO branch costs `nkpts²` evaluations, not `nkpts`

Part I's clause 3 offered two branches — all k-points resident, or "only the
two k-points in flight, recomputing per pair, an `nkpts/2` reduction paid for
in `nkpts`× AO re-evaluation". The memory figure is right. **The cost figure is
wrong**, because `get_k_e1_kpts` is a *double* loop (`fft_jk.py:369` over `k2`,
`:381` over `k1`): with only the pair in flight resident, the inner table is
rebuilt for every outer k, so the cost is **`nkpts²`** AO evaluations, not
`nkpts`. At 3×3×3 that is 729 full-grid `deriv = 1` evaluations against 27.

**Ruled (clause 8; 18-04 Task 2):** one parameter, not two branches. Hold `m`
k-point tables, `m` chosen from `PYSCF_MAX_MEMORY`; tile the inner `k1` loop in
chunks of `m`. Peak `(m+1) · 4 · ngrids · nao` complex, cost `nkpts²/m` AO
evaluations. `m = nkpts` **is** the resident branch and `m = 1` is the worst
case — the two branches Part I asked for are the endpoints of this one number,
and CI still pins a low budget so a small `m` runs.

### 7.3 Clause 9 — `coulG` is rebuilt `nkpts²` times for `nkpts` distinct arguments

`fft_jk.py:384`, inside the double loop:
`coulG = tools.get_coulG(cell, kpt2-kpt1, exxdiv, mydf, mesh)`. On a
Monkhorst–Pack mesh the set `{kpt2 − kpt1}` **is** the mesh, so there are
`nkpts` distinct arguments and `nkpts²` evaluations, each `O(ngrids)` plus the
`exxdiv='ewald'` correction that rides on it.

**Ruled (clause 9; 18-04 Tasks 1 and 3):** key the existing W-01 `coulG` cache
by the k-difference index — the same object the energy path already carries,
not a second one. `nkpts² → nkpts`.

### 7.4 Clause 10 — one image-weighted-contraction primitive, three call sites

Three separate bodies in this phase reduce a `(natm·3, ngrids)` field against a
grid quantity to a `(natm, 3)` array:

* `get_nuc_nuc_grad` (`multigrid_pair.py:916-919`, with upstream's own
  `# TODO improve performance` on it) — 18-09 Task 2;
* `vpploc_part1_nuc_grad` (`pp.py:151-201`) — 18-09 Task 3;
* clause 7's fused `hcore` contraction — 18-05 Task 3.

Part I ruled the first one batched. All three are the same contraction.

**Ruled (clause 10; 18-09 Task 2, consumed by 18-05):** write it once, in
`pyscf-kernels`, with one `oracle_sum` discipline and one determinism test.
Three implementations of one reduction that are supposed to agree, and drift,
is the failure mode GRAD-10 exists to prevent.

### 7.5 Clause 11 — two free ones in `get_k_e1_kpts`

* **`vR_dm *= expmikr.conj()` (`:402`) runs over the whole
  `(3, nset, nao, ngrids)` array — 30 MiB on diamond `gth-dzvp` — even when
  `expmikr` is `np.array(1.)`** (`:386-387`, the `is_zero(kpt1-kpt2)` branch).
  Skip it on the diagonal: `nkpts` full-array complex multiplies, free.
* **The 18 displaced cells are built twice**, once by `get_ovlp` and once by
  `get_kin` (`rks_stress.py:90` and `:103`). Under clause 2 the production path
  builds none; the **gate** still builds them, and it builds each pair once and
  runs both operators on it. 36 → 18 in the test.

---

## 8. Memory — three corrections to Part I's own numbers

### 8.1 Clause 12 — the `get_k_e1_kpts` inner peak is ~2× §3.1's figure, and upstream's `/4` is not slop

`fft_jk.py:389-394`:

```python
rho1 = np.einsum('aig,jg->aijg', ao1T[1:,p0:p1].conj()*expmikr, ao2T)
vG = tools.fft(rho1.reshape(-1,ngrids), mesh)     # rho1 STILL ALIVE
rho1 = None
vG *= coulG
vR = tools.ifft(vG, mesh).reshape(...)            # vG STILL ALIVE
```

`rho1` and `vG` coexist across the forward transform; `vG` and `vR` across the
inverse. **The peak is ~2× the `rho1` figure** — §3.1's 755 MiB is really
~1.5 GiB on diamond `gth-dzvp`, and its 116 MiB tagged is ~232 MiB.

And this reframes the `blksize` finding. `:363` is
`(max_memory-mem_now)*1e6/16/4/3/ngrids/nao`: the `16` is bytes/complex, the
`3` the component axis, and **the `4` is exactly this buffer multiplicity** —
deliberate headroom, not slop. Part I ruled "re-derive, do not transcribe",
which is right; what it did not say is *which part is the defect*. **Only the
doubled `mem_now` is** (`:362` subtracts it, `:363` subtracts it again).

**Ruled (clause 12; 18-04 Task 2, 18-01 Task 5):** the re-derived `blksize`
keeps a buffer-multiplicity factor and drops the double subtraction. 18-01
measures the true multiplicity (2 if the FFT can run in place, 3–4 if not) and
18-04 uses the measured number, with the derivation in the doc comment.

### 8.2 Two smaller corrections, recorded so they are not rediscovered

* **§3.1's `ao1_kpts` table is not doubled by `ao2_kpts`.** `fft_jk.py:347-349`
  binds `ao1_kpts = ao2_kpts` and then rebinds `ao2_kpts` to `[ao[0] for ...]`
  — **views** into the same deriv-1 buffer. One table, and §3.1's 1.02 GiB at
  3×3×3 stands.
* **The MO factorisation adds a resident array while shrinking a transient
  one.** `:357-359`'s `ao2_kpts = [np.dot(mo_coeff[k].T, ao) ...]` materialises
  a *new* `nkpts · nocc · ngrids` complex array — +42 MiB at diamond
  `gth-dzvp` 3×3×3 — beside the deriv-1 table, which stays alive as
  `ao1_kpts`. Against a ~1.4 GiB cut in the transient it is an overwhelmingly
  good trade, which is exactly why 18-04 Task 3's assertion must be on **peak**
  RSS and the plan must state both sides, or the assertion gets a surprise.

---

## 9. Part I §4.3 closed: the strain AO family is **not** algebraically redundant

Part I recorded, and deliberately did not pursue: *"if the whole `ao_strain`
reduced to `ao_deriv(n+1) ⊗ coords`, 18-11 would not be needed at all …
recorded so that a later session can test it cheaply."* This session tested it,
on paper, and the answer is **no** — with a proof rather than the "weak
evidence" Part I had.

`test_eval_ao_grid_response` (`:196-240`) shows the **grid** response *is*
`ao_deriv ⊗ coords`. The **basis** response is not, and clause 2 is why: the
strain derivative of a periodic AO carries the per-image weight `(R_A + L)_y`
*inside* the lattice sum, and `pbc_eval_gto` returns a table that has **already
summed over images**. The weight cannot be recovered from the summed table.

The integral route (clause 2) escapes this because the image sum can be
re-run with a weighted phase; the collocation route cannot, because the
evaluation and the sum are the same pass. That is precisely why upstream ships
a dedicated C kernel (`grid_ao.c:431`) for the AOs and an ordinary integral for
the overlap.

**18-11 stands, and §4.3 is answered rather than open.** Record the reasoning
in 18-11's module doc so the question is not reopened a third time.

---

## 10. Summary of D-PBC-31

| clause | ruling | plan | axis | size of the win |
|---|---|---|---|---|
| 1 | Gate A is **A1 1e-9 / A2 2e-9 / A3 1e-8**, not a flat 1e-8 | 18-CONTEXT, 18-01, 18-11, 18-12, 18-13, 18-15 | precision | 10× tighter on 13 of 14 component gates, at zero cost |
| 2 | `get_ovlp`/`get_kin` ship the **closed form**; FD is the oracle only | 18-02, 18-12, 18-13 | all three | 36 lattice sums → 2; 36 cell builds → 0; removes truncation, cancellation and screening error entirely |
| 3 | the strain FD transforms k at fixed **fractional** coordinates | 18-02, 18-13 | precision | prevents a wrong derivative that is correct at Γ |
| 4 | `.real` moves **inside** the contraction | 18-05 (+06/07/08/10) | speed | half the multiplies, real accumulator, bit-identical |
| 5 | FD divides by the **realised** step | 18-02 | precision | ~2e-12 — free, and explicitly **sub-floor** |
| 6 | `coulG_1` 9→6 by symmetry; `weight_1` is diagonal | 18-12 | speed/precision | 1.5× on 1.7–15.5 MiB; makes two exact facts explicit |
| 7 | `hcore_deriv` contracts against the **density in G-space**; never materialises the matrix | 18-05 | speed/memory | `ngrids·nao²` work `natm·3`× down; `natm·3` iFFTs → 0; subsumes D-PBC-30 clause 6 |
| 8 | the AO cache is one budgeted `m`, tiled; streaming costs `nkpts²/m`, not `nkpts` | 18-04 | speed/memory | corrects D-PBC-30 clause 3's cost model by `nkpts`× |
| 9 | `coulG` cached by k-difference | 18-04 | speed | `nkpts² → nkpts` |
| 10 | one `(natm,3) ← (natm·3, ngrids)×(ngrids)` primitive, three call sites | 18-09, 18-05 | speed | one implementation, one determinism test |
| 11 | skip the identity phase multiply; build the 18 gate cells once | 18-04, 18-12 | speed | free |
| 12 | `blksize` keeps the **buffer multiplicity**, drops the doubled `mem_now`; the inner peak is ~2× §3.1 | 18-01, 18-04 | memory | corrects a 1.5 GiB peak sized at 755 MiB |

**Corrections to Part I:** §3.3's "1e-8-order discontinuity" is an upper bound,
not a realised win (§6.2); §3.4's "`nkpts`× AO re-evaluation" is `nkpts²`
(§7.2); §3.1's 755 MiB inner peak is ~1.5 GiB (§8.1). §3.1's 1.02 GiB
`ao1_kpts` figure is confirmed. **Part I §4.3 is closed** (§9).

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

3. **The strain AO array may be algebraically redundant.**
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
| 2 | one `Ls` for all 36 strain lattice sums, via `intor_cross_with_images` | 18-12 | 36 → 1 `get_lattice_ls`; removes a 1e-8-order FD discontinuity |
| 3 | k-point AO table is a budgeted cache; both branches CI-exercised; re-derive `blksize` | 18-04 | 1.02 GiB → 76 MiB peak at 3×3×3, paid in recompute |
| 4a | tagged `mo_coeff`/`mo_occ` density; factorised `get_k_e1_kpts` | 18-02, 18-04 | inner `rho1` 755 → 116 MiB (`gth-dzvp`) |
| 4b | gradient route **refuses** `kk_symmetry`, with a test | 18-04 | prevents a silent last-bit corruption |
| 5 | fuse strain + ordinary AO into one kernel pass **iff 18-01 measures a win** | 18-01, 18-11 | one pass over a 46-component buffer; unmeasured |
| 6 | hoist the AO table out of `hcore_deriv`'s atom loop, under clause 3's cache | 18-05 | `natm`× fewer full-grid AO evaluations |
| 7 | fuse the stress XC contraction into the block loop; `oracle_sum` the accumulator | 18-12 | 4–5× on a 6.7–59 MiB array |

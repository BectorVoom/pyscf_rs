# KUKS Speed & Precision Optimisation Plan — `pyscf-pbc-dft`

**Created:** 2026-08-31
**Target:** `pyscf_pbc_dft::kuks::Kuks` (`pbc.dft.KUKS`) and the `nset = 2` path
beneath it. Sibling to [`KRKS-OPTIMISATION-PLAN.md`](./KRKS-OPTIMISATION-PLAN.md),
whose §1.2 explicitly excluded `KUKS` — *"They share `KNumInt` and `fft_jk`, so
they inherit the wins for free, but their drivers are not touched."* This plan is
the driver.
**Status:** draft — no code written. **§2.2 (the open-shell divergences) is
VERIFIED against both sides of the comparison. §2.1 (the cost model) is
MODELLED, NOT MEASURED, and its inherited premise is STALE** — see §2.1.0.
**Audience:** an execution agent that follows instructions literally and does NOT
infer.

---

## 0. HOW TO EXECUTE THIS PLAN

Inherits every standing rule of [`PBC-MASTER-PLAN.md`](./PBC-MASTER-PLAN.md) §0
and of `AGENTS.md`, and every rule of `KRKS-OPTIMISATION-PLAN.md` §0 — RULE 4
(tests in separate files, never `mod tests` in a production source file), RULE 5
(cubecl: read the manual first, `<F: Float + CubeElement>`, on any build error
read `cubecl_error_guideline.md` **before** touching the code), RULE 6 (the
ALG-06 dependency wall), RULE O (measure, change ONE thing, re-measure).

Two rules are specific to this plan.

* **RULE U — no KUKS work item may be validated on a closed-shell cell.**
  On a closed-shell cell `dm_a == dm_b` **bit-identically and permanently**
  (§2.2.1 proves it is an exact fixed point of this port's SCF map), so the
  unrestricted path degenerates to the restricted one and a passing test proves
  nothing about it. Every gate in this plan runs on a cell where
  `dm_a != dm_b`. This is why **U-00 comes first and blocks everything else.**

* **RULE V — the plan's own numbers are evidence-tagged.** Every quantity below
  carries `MEASURED (source)`, `MODELLED` or `UNVERIFIED`. Do not promote one to
  another without doing the work. Where this plan corrects its KRKS sibling, it
  says so as an explicit erratum, the way that plan's §8 Q3 does against
  `PBC-MASTER-PLAN` plan 11-01.

---

## 1. Scope and the gates

### 1.1 In scope

| stage | code | crate |
|---|---|---|
| the KUKS driver | `kuks.rs:190-270` `veff_from_parts`, `:359-384` `energy_elec` | `pyscf-pbc-dft` |
| the open-shell numerical integration | `numint.rs:345-415` `nr_uks` | `pyscf-pbc-dft` |
| the `nset = 2` J/K dispatch | `veff.rs:38-121` `get_jk`, `:147-173` `trace_dm_v`/`trace_ab` | `pyscf-pbc-dft` |
| the `nset = 2` Coulomb build | `fft_jk.rs:63-115` `get_j_kpts` | `pyscf-pbc-df` |
| the unrestricted init-guess / occupation surface | `init_guess.rs:38-128`, `kocc.rs:49-82` | `pyscf-pbc-scf` |

`Kuks::veff_from_parts` is `pub` and `self`-free precisely so `Kroks` can call
the identical code (`kroks.rs:119-127`), so **every change to it is inherited by
KROKS** and must be tested there too.

### 1.2 Out of scope (non-goals)

* Everything `KRKS-OPTIMISATION-PLAN.md` owns: the FFT (W-02/W-02b, **landed**),
  the `fft_jk` reductions (W-05, **landed**), the `coulG`/`expmikr` hoist (W-01,
  **landed**), and W-03, W-04, W-06, W-07, W-08, W-09 (**unstarted**). Where an
  item here depends on one of those, it says so and waits.
* `Kukspu` — `kspu.rs:679-716` has **no `impl KOverrideHooks`**, so DFT+U/UKS
  cannot be driven by `kscf::kernel` at all. Recorded as U-08; a completeness
  gap for the DFT+U owner, not an optimisation.
* `kuks_ksymm` (Phase 17), `pbc/grad/kuks` (Phase 18), `pbc/tdscf/kuks` (Phase 19).

### 1.3 GATE A — the existing accuracy gate must not regress

`crates/pyscf-pbc-dft/tests/gate.rs` at its **current** tolerances, unchanged.
The KUKS row: `KUKS Si 2×2×2 PBE` (GTH-pade) at `1e-11`, last residual
**~6.45e-12** — MEASURED (`SUMMARY.md:129`).

```bash
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture
```

**Read RULE U before trusting this gate.** `gate.rs:266-290` runs on `silicon()`,
which is closed-shell. It exercises `nr_uks`, the `vj[0]+vj[1]` sum and
`sub_scaled(.., 1.0, vk)` — real code — but on inputs where the two spin channels
are bit-identical. It cannot see anything in §2.2.

### 1.4 GATE B — determinism must not regress

D-PBC-17: every complex reduction reaching an energy, a density matrix or a
convergence test goes through the ordered primitives, and the result is
bit-identical under `RAYON_NUM_THREADS=1` and `=8`. Already verified for the
KUKS row — MEASURED (`SUMMARY.md:118-127`: all six gate energies bit-identical
across thread counts 1/8/16).

### 1.5 GATE C — no FMA contraction (FOUND-05)

`cargo run -p xtask --bin check-no-fma`. `SCAN_TARGETS` is currently
`[pyscf-algebra, pyscf-core, pyscf-ccsd]`. No item in this plan adds a cubecl
kernel, so no item here is required to extend it; U-05 inherits KRKS W-06's
obligation to add `pyscf-pbc-dft`.

### 1.6 GATE U — the open-shell gate (NEW; U-00 builds it)

**This gate does not exist today and nothing in §2.2 can be fixed before it
does.** KUKS against live upstream on cells where `dm_a != dm_b`:

| case | cell | k-mesh | tolerance | what only it can see |
|---|---|---|---|---|
| U-a | `li_atom_spin1()`, all-electron | Γ and `[1,1,3]` | **1e-12** | the `spin != 0` path; no GTH floor |
| U-b | `h2_stretched_spin0()`, all-electron | Γ | **1e-12** | `_break_dm_spin_symm` (§2.2.1) |
| U-c | either, `xc = pbe0` | Γ | **1e-11** | the doubled K contractions, `sub_scaled(.., 1.0, vk)` |

All-electron deliberately: the GTH cells floor at ~4e-12 for structural reasons
inherited from `get_pp`, and the all-electron control is what proves 1e-12 is
reachable (`KRKS He-fcc` sits at 9.81e-14). **Do not gate the open-shell work on
a pseudopotential cell.**

---

## 2. Where the time and the precision go

### 2.1 The cost model

#### 2.1.0 READ THIS FIRST — the inherited baseline is STALE

`KRKS-OPTIMISATION-PLAN.md` §2.1 MEASURED, on Si `gth-szv` (`nao = 8`), 2×2×2,
`mesh = 21`, PBE, **pre-W-02**:

| stage | wall time | share of a hybrid `get_veff` |
|---|---|---|
| `nr_rks` (warm AO cache) | 22.7 ms | 0.3 % |
| `get_j_kpts` | 13.9 ms | 0.2 % |
| `get_k_kpts` | 6.60 s | **99.5 %** |
| — of which the 3-D transform | 5.59 s | **93 % of `get_k_kpts`** |

**Those shares no longer hold.** `SUMMARY.md:58-140` records that **W-02
(mixed-radix + Rader) and W-02b (rayon over the transform batch) both landed**,
while **W-03 (GEMM/device contractions) did not start**. So the transform got
both algorithmically cheaper (§2.3's table: 2.1×–3.0× fewer operations at
PBC-relevant mesh sizes) and parallel across cores, while every contraction in
`fft_jk.rs` remains a scalar single-threaded loop. The transform's share of
`get_k_kpts` has fallen by a large but **unmeasured** factor, and the
contraction share — the part that KUKS doubles — has risen correspondingly.
`SUMMARY.md:6-13` says as much directly: that session ran on a contended machine
and *"no throughput number in this summary should be read as a clean
measurement."*

**Consequence, and it is binding: this plan does not quote a KUKS/KRKS
multiplier. U-01 measures it.** Any number in §2.1.2 below is MODELLED.

#### 2.1.1 What actually doubles — VERIFIED code structure

Each row verified by reading the source in this session.

| fact | evidence |
|---|---|
| `get_k_kpts`'s transform operand `rho1` is built from `(ao1t, ao2t, expmikr, p0, p1, nao, ngrids)` — **no density argument** | `fft_jk.rs:288`, `build_rho1` at `:395` |
| `fft` (`:291`), the `coulG` multiply (`:292-299`), `ifft` (`:300`) and the `real_out` truncation (`:302-306`) are all **outside** `for i in 0..nset` | `fft_jk.rs:288-306` |
| the `nset` loop wraps only `contract_vr_aodm` (`:309-313`), the `expmikr.conj()` sweep (`:317-330`) and `accumulate_vk` (`:333-337`); `dm_times_conj_ao` is per-`k2` inside `dms.iter().map` (`:246-250`) | `fft_jk.rs` |
| `blksize` comes from `df.max_memory / 16 / 4 / ngrids / nao` — **`nset` does not appear** | `fft_jk.rs:238-239` |
| ⇒ **KRKS and KUKS issue the same number of transforms of the same batch shape.** Only the contractions double. | — |
| `get_j_kpts` puts **everything** inside `for dmset in dms.iter()`: `accumulate_rho` over all k, the `1/nkpts` scale, `fft`, the `coulG` multiply, `ifft`; the band contraction is a second `nset` loop | `fft_jk.rs:63-88`, `:102-110` |
| — but `get_gv`, `get_coulg` and `df.ao_kpts` are hoisted above it and are shared | `fft_jk.rs:45-55` |
| `nr_uks` calls `eval_ao` **once** per block (shared), then `eval_rho` twice and `accumulate_vxc` twice | `numint.rs:377` vs `:384-385`, `:396-399` |

**The single most important structural fact in this plan:** the expensive half of
`get_k_kpts` is spin-independent. A hybrid KUKS is **not** 2× a hybrid KRKS.
This is an erratum against the natural assumption, and it is why this plan's
speed items are small and its precision items are the headline.

#### 2.1.2 The multiplier — MODELLED, bounds only

Not all of the non-transform work doubles. By FMA count `build_rho1` (~2 complex
multiplies × `nao²·Ng` per pair) and the `coulG` sweep are the same order as
`contract_vr_aodm` + `accumulate_vk` combined, and neither doubles. So roughly
**half** of the non-transform work is nset-dependent.

* **Hybrid.** On the stale pre-W-02 shares the ceiling is 1.075× and the
  midpoint ~1.05×. On a post-W-02 baseline the transform share is much smaller
  and the multiplier is correspondingly larger — a MODELLED illustration, not a
  measurement: if the transform got 2.1× from radix and ~5× from rayon, its
  share falls to ~34 % and the multiplier rises toward **~1.3×**.
  **Bound the claim at `1 < multiplier < 2` and let U-01 close it.**
* **Pure functional.** `nr_uks` ≈ 2× `nr_rks` on the contractions (AO evaluation
  shared, one `eval_xc_eff_uks` call on a 2×-wider kernel), and J is just under
  2×. **"Just under 2×" — not "2×".**

**Memory, so a reader does not assume it doubles.** In `get_k_kpts` the
nset-dependent buffers are `ao_dms` + `vr_dm` = `2·nset·nao·ngrids`; the
nset-independent ones are `rho1`, `vg`, `vr` = `3·nao²·ngrids`. At `MESH_GATE`
(`ngrids = 29791`, `nao = 8`) that is ~15 MiB against ~91 MiB, so KUKS peak RSS
in `get_k_kpts` is roughly **+15 %**, not +100 %.

**Allocation churn, which grows in relative terms as the transform shrinks.**
`fft_jk.rs:277-279` allocates `nset` fresh `CTensor::zeros(nao·ngrids)` per
`(k2,k1)` pair — 64 pairs for the gate cell, 3.81 MiB each ⇒ **488 MiB of
allocate-and-zero per `get_k_kpts` for KUKS against 244 MiB for KRKS**.
Negligible against a 6.6 s call; **1–3 % against a post-W-02 call in the 1–2 s
range**, i.e. it stops being noise exactly where this plan is aimed. Hoisting
the allocation out of the pair loop and zeroing in place removes it (U-06).

#### 2.1.3 Two attribution notes — NOT KUKS defects, but inside "the 7 %"

Recorded so U-01 does not misattribute them to the spin doubling.

* **W-01's cache is partly defeated by a clone.** `fft_jk.rs:270-274` does
  `(entry.0.clone(), entry.1.clone())` on every `(k1,k2)` pair. W-01 hoisted the
  *computation* out of the `Nk²` loop but not the copy. At `MESH_GATE`, `coulg`
  is 233 KiB and `expmikr` 466 KiB ⇒ ~45 MiB of pure memcpy per `get_k_kpts`.
  Fixable by returning a borrow or an `Arc`. **File this against the KRKS plan,
  not here.**
* **`ewald_exxdiv_for_g0` rebuilds density-independent quantities every call.**
  `df_jk.rs:59-68` recomputes `pbc_intor(cell, "int1e_ovlp", kpts, ..)` and
  `madelung(cell, kpts, None)` on every `get_k_kpts`. Its nset-dependent part
  (`df_jk.rs:81-86`, two `zmm_small` per `(set, k)`) is `O(nset·Nk·nao³)` and
  genuinely negligible. Also a KRKS-plan item.

### 2.2 Precision — the open-shell divergences (**the headline**)

This is what the plan is for. None of it is about rounding.

#### 2.2.1 `dm_a == dm_b` is an exact fixed point, and nothing can break it

**Port side.** `init_guess.rs:101` returns, for `nset = 2`:

```rust
vec![half.clone(), half]
```

Not "approximately equal" — the same `Vec<CTensor>`, cloned. The chkfile branch
does the same at `:76`. And there is **no** symmetry-breaking code anywhere in
the workspace: `grep -rn "break_dm_spin_symm\|breaksym\|break_symm" --include=*.rs crates/`
returns **zero matches**.

**Upstream side.**

| line | what it does |
|---|---|
| `pyscf/scf/uhf.py:855-863` | `UHF.init_guess_by_minao` does `dma = dmb = dm*.5`, then calls `_break_dm_spin_symm(mol, (dma, dmb), breaksym)` |
| `pyscf/scf/uhf.py:116-134` | `_break_dm_spin_symm` fires when `breaksym and mol.spin == 0 and abs(dma-dmb).max() < 1e-2`; for `breaksym == 1` it keeps only the **intra-atomic** blocks of `dmb` |
| `pyscf/scf/uhf.py:778` | `init_guess_breaksym = getattr(__config__, 'scf_uhf_init_guess_breaksym', 1)` — **default 1** |
| `pyscf/pbc/scf/kuhf.py:417` | `KUHF` re-declares the same default |
| `pyscf/pbc/scf/kuhf.py:421-425` | `init_guess_by_{1e,minao,atom,huckel,mod_huckel} = pbcuhf.UHF.<same>` — KUHF inherits both the flag and the methods |

**And it is a genuine fixed point, not merely a different starting point.**
Traced through this port's SCF map with `cell.spin == 0`:

1. `dm_a == dm_b` ⇒ `Kuks::veff_from_parts` builds `sets` (`kuks.rs:205`) with
   identical channels ⇒ `eval_xc_eff_uks` on `rho_a == rho_b` ⇒ symmetric `vxc`;
   `vj[0] == vj[1]`, `vk[0] == vk[1]` ⇒ `vhf[0] == vhf[1]` bitwise.
2. `Kuks::eig` (`kuks.rs:319-324`) runs the same deterministic `eig_channel` on
   bit-identical inputs ⇒ bit-identical `(e, c)`.
3. `Kuks::get_occ` (`kuks.rs:340`) → `get_occ_unrestricted(ea, eb, na, nb)` with
   `ea == eb` and `na == nb` ⇒ `fermi_level` (`kocc.rs:88-105`) sorts the same
   pooled list ⇒ `fermi_a == fermi_b` exactly ⇒ the `<= fermi` tests at
   `kocc.rs:60` and `:71` give `occ_a == occ_b`.
4. `make_rdm1` (`kuks.rs:346-357`) ⇒ `dm_a == dm_b` again.
5. Nothing in the driver perturbs it: `kscf.rs:150-203`'s DIIS, damping and level
   shift are all linear and preserve the symmetry. There is no stability
   analysis and no `newton`.

**The correct statement of the defect is therefore stronger than "the initial
guess differs": this port's KUKS/KUHF is structurally incapable of reaching a
spin-broken solution at `cell.spin == 0`.** Upstream reaches AFM and other
spin-broken minima on exactly these cells **by default**. `KInitGuess::UserDm`
(`init_guess.rs:48`) is the only escape hatch and requires the caller to
hand-build the broken density matrix.

#### 2.2.2 The initial guess is renormalised on one total, not per channel

Upstream, `pyscf/pbc/scf/kuhf.py:476-486`:

```python
ne = lib.einsum('xkij,kji->x', dm_kpts, s1e).real   # LENGTH 2
nelec = np.asarray(self.nelec)                       # (nalpha, nbeta)
if np.any(abs(ne - nelec) > 0.01*nkpts):
    dm_kpts *= (nelec / ne).reshape(2,-1,1,1)        # PER CHANNEL
```

This port, `init_guess.rs:106-126`: `electron_count` (`krdm.rs:110-118`) sums
over **both** sets into one `f64`, and one scale `s = nelectron / ne` is applied
to every set, against `nelectron = (na + nb) as f64`. The *totals* are on the
same footing — `Kuks::nelec()` (`kuks.rs:102-117`) returns BZ-supercell counts
via `cell.tot_electrons(nkpts)`, matching `kuhf.py:442-456` — so that half is
right.

**Where they diverge, and the first case is the serious one:**

1. **`cell.spin != 0` — always, and it removes the only polarisation in the
   guess.** The port always produces `dm_a == dm_b`, so `ne_a = ne_b = Ne/2` and
   `|ne_a - nalpha| = |spin|/2`. Upstream's threshold `> 0.01·nkpts` therefore
   fires for any `|spin| >= 1` and `nkpts <= 50`, and upstream then scales alpha
   by `nalpha/(Ne/2)` and beta by `nbeta/(Ne/2)` — **different factors.** Since
   `_break_dm_spin_symm` short-circuits on `mol.spin == 0`, **this
   renormalisation is the only thing that polarises upstream's minao guess for
   an open-shell cell.** The port applies one factor to both channels and hands
   the SCF a completely unpolarised guess.
2. **`cell.spin == 0` with a broken `dmb`** — upstream renormalises the two
   channels differently to restore `(nalpha, nbeta)`; the port has no broken
   `dmb` to restore. Falls out of §2.2.1.
3. **`cell.spin == 0`, unbroken (the port's actual state)** — the scale *factor*
   agrees, but the **threshold differs by 2×**: with `dm_a == dm_b`,
   `ne_total − Ne = 2(ne_a − nalpha)`, so the port fires at
   `|ne_a − nalpha| > 0.005·nkpts` where upstream needs `> 0.01·nkpts`. A real
   but narrow divergence.

**UNVERIFIED, and U-00 must settle it before U-02 asserts anything about it:**
whether the renormalisation branch fires at all for the standard minao guess on
`silicon()`/`diamond()`. Those use `gth-szv` + `gth-pade`, so the AO basis holds
4 valence electrons/atom while the minao guess is built from an all-electron
minimal basis; whether `default_get_init_guess` renormalises to the pseudo
valence count first decides it. There is a `tracing::debug!` at
`init_guess.rs:110-114` printing `ne_per_cell` and `want` — one gate run under
`RUST_LOG=debug` answers it.

#### 2.2.3 Nothing in the test suite can see any of this

* The **only** oracle gate for KUKS is `gate.rs:266-290`, on closed-shell
  `silicon()`.
* The **only** other KUKS test is `smoke.rs:287-305`, asserting
  `|E_KUKS − E_KRKS| < 1e-9` on closed-shell `diamond()`. Given §2.2.1, this
  test proves the closed-shell collapse and nothing about the open-shell path.
* KUHF adds two more, both also closed-shell: `kscf.rs:174-198` and `:466-482`.
* `grep -rn spin crates/pyscf-pbc-dft/tests/ crates/pyscf-pbc-scf/tests/` finds
  one hit, a doc-comment word at `smoke.rs:288`. **No test sets a spin.**

**And the harness cannot express one.** Rust side is ready:
`MoleBuildArgs.spin: i32` exists (`pyscf-gto/src/types.rs:127`, default 0),
`CellBuildArgs.mole` carries it, and `Kuks::nelec()` already reads
`cell.mol.spin` (`kuks.rs:107`) — but `bohr_cell` (`common/mod.rs:69-82`) uses
`..Default::default()` and takes no spin argument. Oracle side is not: `ORACLE_PY`
(`gate.rs:54-104`) unpacks exactly **ten** positional arguments and never sets
`c.spin` or `c.charge`, and `cell_args` (`common/mod.rs:151-162`) serialises only
`a`, `xyz` and `sym`. Both must be extended before any open-shell gate exists.

#### 2.2.4 No spin-contamination diagnostic

`spin_square` (`kuhf.py:591`) is not ported — `grep` is clean across
`pyscf-pbc-scf` and `pyscf-pbc-dft`. Without `<S²>` a spin-contaminated KUKS
solution is indistinguishable from a correct one: **"converged" is not
"correct".** (`canonical_occ` *is* ported, `addons.rs:98-106`.)

#### 2.2.5 `Smearing` has no `fix_spin` — a missing feature, not a defect

`Kuks::get_occ` (`kuks.rs:334-339`) pools all `2·nkpts` energy lists, fills
`na + nb` electrons at `mo_occ_max = 1.0`, and returns one shared Fermi level.
That is **exactly** upstream's `fix_spin=False` branch
(`pbc/scf/smearing.py:107-142`), so it is correct against the default. But
`scf/smearing.py:25` exposes `fix_spin` and `:70-102` implements the per-channel
Fermi level, while this port's `Smearing` (`smearing.rs:34-41`) has only
`sigma`, `method` and `mu0`. **An open-shell KUKS with smearing cannot hold its
spin.** Record it; do not call it a bug.

### 2.3 Precision — reductions on the KUKS energy path

KRKS W-05 fixed `fft_jk`'s reductions and **only** those — `SUMMARY.md:14-16`
names one file in one crate. The following are on the KUKS energy path and were
never in scope of any landed item.

* **`trace_ab` exists TWICE, as two independent naive duplicates, and both reach
  an energy.**
  * `veff.rs:161-173` — naive `sr += ...; si += ...` over `i,j in 0..n`. The
    file imports only `CTensor` (`:16`); `grep oracle_ veff.rs` is empty. Reached
    through `trace_dm_v` (`:147-158`) from `kuks.rs:248` (**ecoul**) and
    `kuks.rs:258` (**exc**), and from `krks.rs:193`/`:203`.
  * `krdm.rs:48-60` — a textually identical function in a **different crate**,
    also un-oracled (`krdm.rs:4` imports only `CTensor`). Reached from
    `Kuks::energy_elec` (`kuks.rs:379`, → **e1** → **e_tot**), `kroks.rs:225`,
    `krks.rs:343`, `kspu.rs:672`. `Kgks` uses the `veff.rs` one
    (`kgks.rs:29,192,285`).
  * **Fixing one leaves `e1` on the naive path. Both must be fixed.**
  * **State the error bound correctly.** This is *not* one
    `nset·Nk·nao²`-long chain. It is `nset·Nk` independent naive chains of
    length `nao²` — each `trace_ab` call starts from a fresh `sr = 0.0` — fed
    into one naive outer chain of length `nset·Nk` in `trace_dm_v`
    (`veff.rs:148-156`). KUKS doubles the *number* of inner chains and the
    *length* of the outer chain, so the worst-case bound grows as
    `O(nset·Nk + nao²)·ε`, **not** `O(nset·Nk·nao²)·ε`. Both sums are
    un-oracled; both need fixing.

* **`nr_uks` and `nr_rks` accumulate across grid blocks with a naive `+=`.**
  `numint.rs:391`, `:392`, `:395` (KUKS) and `numint.rs:315`, `:318` (KRKS) sit
  inside the `for (p0,p1) in self.block_ranges(..)` loops opened at `:374` and
  `:302`. `block_ranges` (`:195-211`) derives `blksize` from `max_memory`,
  `ty.ncomp()` and `nkpts` — **not** from `nset` — so KRKS and KUKS get the same
  partition for a given cell, but **`excsum` and `nelec` are not bit-stable
  across `max_memory`.** This is the precondition KRKS W-07 named for itself and
  never got.

* **A KUKS-only association divergence from upstream.** `numint.rs:395` folds
  two block sums into one statement:
  ```rust
  excsum[i] += oracle_sum(&ta) + oracle_sum(&tb);
  ```
  Upstream `pbc/dft/numint.py:485-486` does **two separate** accumulations:
  ```python
  excsum[i] += dena.dot(exc)
  excsum[i] += denb.dot(exc)
  ```
  `(E + (Sa + Sb))` versus `((E + Sa) + Sb)` — ~1 ulp per block, ~1e-15 Ha
  against a 1e-11 gate, so harmless. But it is a real KUKS-only bit-parity
  divergence from the oracle and it belongs on the record.

* **`KNumInt::eval_rho`'s k-point accumulation is the one KUKS actually
  doubles.** `numint.rs:254-259` does `rho.add_assign(&block)` over
  `for k in 0..nkpts` — an `Nk`-long naive accumulation of `Ng`-vectors, called
  **twice per set** in `nr_uks` (`:384-385`) against once in `nr_rks` (`:311`).
  `Nk`-sized, so small — the same size class W-05 deliberately left alone.

#### 2.3.1 REFUTED — `eval_rho_one` has no grid-length reduction. Do not "fix" it.

An earlier draft of this plan claimed `eval_rho_one` (`numint.rs:820-889`)
reduces over the grid with a naive `+=` while its sibling `vxc_mat_one` uses
`oracle_sum`. **That is wrong, and it is recorded here so no future agent
re-derives it.**

* `vxc_mat_one` (`numint.rs:811-812`): `oracle_sum(&terms_re)` where `terms_re`
  has length `ngrids`. The reduction axis **is** the grid. Correct as written.
* `eval_rho_one` (`numint.rs:864-887`): the loop is `for j in 0..nao { for g in
  0..ngrids { acc_re[g] += ... } }`. **`g` indexes the output**, `j` is the
  reduction axis. Each `acc_re[g]` is a naive sum of exactly `nao` terms. Same
  shape at `:842-858`, where `c0_re[jb + g]` reduces over `i in 0..nao`.

This is precisely the mistake the KRKS plan made about `accumulate_rho` and
which `SUMMARY.md:26-35` refused to act on: *"neither is `ngrids`-sized … the
plan's phrase does not correspond to any actual O(ngrids) sequential sum in this
function."* **There is no defect here. Leave it alone.**

### 2.4 Allocation churn specific to `nset = 2`

| site | what |
|---|---|
| `kuks.rs:205` | `[vec![dms[0].clone()], vec![dms[1].clone()]]` — clones **both** full spin DM k-stacks on every `get_veff`, purely to reshape for `nr_uks` |
| `kuks.rs:209` | `.clone()` on `nr.vmat[..][0]`, an **owned** `NrKUksResult` — should be a move |
| `kuks.rs:246-251` | `vec![jtot.clone(), jtot.clone()]` — two full Coulomb k-stacks cloned and dropped every `get_veff`, only to satisfy `trace_dm_v`'s `v[s][k]` shape |
| `kuks.rs:148-159` | `get_rho` allocates a whole new k-stack to spin-sum |
| `fft_jk.rs:277-279` | `vr_dm` sized `nset`, allocated per `(k2,k1)` pair — §2.1.2's 488 MiB |
| `numint.rs:367-372` | the `vmat` zero-stacks allocated fresh every SCF cycle |
| `numint.rs:381-399` | per-block `rho_a/rho_b/dena/denb/ta/tb/wv/out`, all fresh, **twice** for KUKS |
| `xc.rs:367`, `:613` | the XC string is re-parsed **per grid block**, twice |
| `veff.rs:48-49` | and twice more per `get_veff` |

### 2.5 A shared defect that is NOT KUKS-specific — do not claim it as one

`Kuks` does not override `get_grad`, so with smearing enabled it uses the
occupied-virtual block from the `KOverrideHooks` default
(`khooks.rs:148-166`) rather than `smearing::grad_tril`. `Krhf::get_grad`
(`krhf.rs:286-315`) *does* branch on `self.smearing`, citing
`pbc/scf/smearing.py:25-31`. But `grep -rn "fn get_grad"` shows overrides only in
`krohf.rs:388`, `krhf.rs:286` and `kroks.rs:194` — so `Krks` (`krks.rs:71`),
`Kuks` (`kuks.rs:52`) and `Kuhf` (`kuhf.rs:37`) all support smearing and all use
the wrong criterion. **This is a three-driver shared defect. File it against the
SCF driver family, not against KUKS.**

---

## 3. Work items

Numbered `U-nn` so they never collide with the KRKS plan's `W-nn`. Ordered.
Each is independently landable and independently revertible.

---

### U-00 — The open-shell fixture and GATE U (**do this first; it blocks §2.2 entirely**)

**FILES** `crates/pyscf-pbc-dft/tests/common/mod.rs`,
`crates/pyscf-pbc-dft/tests/gate.rs`

**WHY** §2.2.3. RULE U. No fix in U-02 can be shown correct without a test that
can see `dm_a != dm_b`, and no such test can exist until the oracle harness can
express a spin.

**STEPS**

1. Add `spin: i32` to `cell_args` (`common/mod.rs:151-162`) and an eleventh
   positional argument to `ORACLE_PY` (`gate.rs:54-104`), setting `c.spin`
   before `c.build()`. Add `charge` at the same time — the same gap, the same
   one-line fix, and an open-shell ion test will want it.
2. Give `bohr_cell` (`common/mod.rs:69-82`) a spin parameter, or add a
   `spin_cell` sibling; today it hard-codes `..Default::default()`.
3. Set `mf.init_guess_breaksym` **explicitly** in `ORACLE_PY` so the gate states
   which guess it is measuring rather than inheriting a default that may move.
4. Emit `mf.spin_square()` in the oracle payload — U-07 needs it and adding it
   now costs nothing.
5. Two fixtures:
   * `li_atom_spin1()` — Li in a cubic box, all-electron `sto-3g`, `spin = 1`.
     Genuinely open-shell, 5 AOs, no GTH floor.
   * `h2_stretched_spin0()` — H₂ at ~3 Bohr, all-electron `6-31g`, `spin = 0`.
     The spin-symmetry-**breaking** case; the only thing that can see U-02.
6. Wire GATE U's three rows (§1.6) over `{lda,vwn, pbe, pbe0}`.
7. **Answer §2.2.2's UNVERIFIED question in the same pass:** run the existing
   gate under `RUST_LOG=debug` and record whether `init_guess.rs:110-114` fires
   on `silicon()`/`diamond()`. Write the answer into this document.

**THE K-MESH PARITY TRAP — document it in the test file.** `Kuks::nelec()`
(`kuks.rs:102-117`, port of `kuhf.py:442-458`) forms
`nalpha = (ne_supercell + spin) / 2` where `ne_supercell = cell.tot_electrons(nkpts)`
but `spin` is **per cell**. An odd-electron cell with an even k-count therefore
fails `nalpha + nbeta == ne` and is rejected — by upstream *and* by this port,
identically. Use Γ or an odd k-count (`[1,1,3]`). **This is an inherited
upstream constraint, not a port bug; do not "fix" it.**

**BIT-PARITY** n/a — additive.

**DONE** GATE U green on the *current* code for U-a and U-c (they measure the
`spin != 0` path, which §2.2.2 says is unpolarised — **so U-a may FAIL on
landing, and that failure is the finding**). U-b is expected to fail until U-02.
**Record both residuals; do not weaken a tolerance to make a row pass.**

**RISK** medium. A small open-shell cell may not converge. Choose the fixture
for convergence, record the `KScfConfig` used, and if `li_atom_spin1` is
recalcitrant try a closed-box atom with a larger vacuum before changing the
tolerance.

---

### U-01 — The KUKS profiling harness (**the speed prerequisite**)

**FILES** `crates/pyscf-bench/src/bin/krks_profile.rs` (extend; consider renaming
to `pbc_ks_profile.rs`)

**WHY** §2.1.0. Every number in §2.1.2 is MODELLED and the premise it inherits
is pre-W-02. RULE O forbids optimising against a stale profile.

**STEPS**

1. Add `--driver {krks,kuks}`. The harness is KRKS-only today
   (`krks_profile.rs:28` imports `pyscf_pbc_dft::krks`).
2. **The cheapest useful measurement needs no new cell and no new physics:**
   `krks_profile.rs:160-171` already calls `get_k_kpts(&df, &result.dm, ..)`
   directly, so passing `&[result.dm[0].clone(), result.dm[0].clone()]` gives
   the `nset = 2` timing against the `nset = 1` timing on identical data. Do
   this first — it answers §2.1.2's multiplier on its own.
3. Split `nr_rks`/`nr_uks` and `get_j_kpts` out as separately timed stages; today
   only `get_j_kpts`/`get_k_kpts` and a full `kernel()` are timed.
4. Re-measure the transform share of `get_k_kpts` **post-W-02**, and record it
   here as an erratum against `KRKS-OPTIMISATION-PLAN.md` §2.1's 93 %.
5. Add the U-00 cells once they exist.
6. Commit a baseline JSON. The KRKS plan's W-00 DONE criterion was never met
   (`SUMMARY.md:110-116`: *"a working harness, not a committed baseline"*) —
   meet it here, on an idle machine.

**DONE** §2.1.2's `1 < multiplier < 2` replaced by a measured number, for both a
pure functional and a hybrid, at `mesh ∈ {21, 31}`.

**RISK** none — additive.

---

### U-02 — **PRECISION HEADLINE:** `_break_dm_spin_symm` + per-channel renormalisation

**FILES** `crates/pyscf-pbc-scf/src/init_guess.rs`,
`crates/pyscf-pbc-scf/src/kuhf.rs`, `crates/pyscf-pbc-dft/src/kuks.rs`,
`crates/pyscf-gto/src/` (see the layering step)

**WHY** §2.2.1 and §2.2.2. This port cannot reach a spin-broken solution at all,
and hands an unpolarised guess to every `spin != 0` cell. Everything else in this
plan is a rounding-level concern by comparison.

**STEPS**

1. Port `_break_dm_spin_symm` (`uhf.py:116-134`) **line by line** (RULE 2),
   including the `breaksym == 2` branch and the `abs(dma-dmb).max() < 1e-2`
   guard. Do not invent a different symmetry-breaking scheme.
2. Add `init_guess_breaksym: i32` to `Kuhf` and `Kuks`, default `1`, matching
   `__config__.scf_uhf_init_guess_breaksym` (`uhf.py:778`, `kuhf.py:417`).
3. Replace `init_guess.rs:106-126`'s single total renormalisation with the
   per-channel form of `kuhf.py:476-486`: a length-2 `ne`, compared against
   `(nalpha, nbeta)`, with a per-channel scale. `electron_count`
   (`krdm.rs:110-118`) currently sums both sets into one `f64` and needs a
   per-set sibling.
4. Correct the false doc comment at `init_guess.rs:30-33`, which asserts the
   halved restricted density "is the same matrix" as upstream's UHF guess.
   It is not — §2.2.1.
5. **A layering sub-step this item must budget for.** `_break_dm_spin_symm`
   needs `aoslice_by_atom`, which today lives at **`pyscf-grad/src/rhf.rs:506`**
   — the gradients crate. `pyscf-pbc-scf` must not depend on `pyscf-grad` (SCF
   sits below gradients), so the function moves to `pyscf-gto` next to `Mole`
   and both call sites re-point. Verify against
   `cargo run -p xtask --bin check-dependency-wall`.

**BIT-PARITY** **NO on any open-shell cell — that is the entire point.**
**Must be EXACT on closed-shell `silicon()`**: `_break_dm_spin_symm`'s
`mol.spin == 0` branch does fire there, so verify explicitly that the guard
`abs(dma-dmb).max() < 1e-2` and the subsequent SCF still land on the same
energy — the unmoved GATE A residual (~6.45e-12) is the regression check.

**TEST** GATE U rows U-a and U-b. Plus a unit test in
`crates/pyscf-pbc-scf/tests/init_guess_spin.rs` asserting the guess itself:
`dm_a != dm_b` for `h2_stretched_spin0`, and per-channel electron counts equal
to `(nalpha, nbeta)` to 1e-12 for `li_atom_spin1`.

**DONE** GATE U green for U-a and U-b; GATE A residual unmoved.

**RISK** high — this changes which SCF solution is found. Its "regression" is
only detectable against the **new** reference, not the old one, because the old
one converges to a different (higher) stationary point. Land U-00 first and land
nothing else alongside it.

---

### U-03 — **PRECISION:** ordered reductions on the KUKS energy path

**FILES** `crates/pyscf-pbc-dft/src/veff.rs`,
`crates/pyscf-pbc-scf/src/krdm.rs`, `crates/pyscf-pbc-dft/src/numint.rs`

**WHY** §2.3. A standing D-PBC-17 gap that W-05 never covered.

**STEPS**

1. Route **both** `trace_ab` duplicates — `veff.rs:161-173` and `krdm.rs:48-60`
   — through `pyscf_algebra::oracle_zdot` (`zoracle.rs:36`) or `oracle_dot`
   (`oracle.rs:30`), the way W-05 did in `fft_jk`. Fixing one leaves `e1` naive.
   **Better: delete one of them.** Two textually identical un-oracled copies in
   two crates is how they drifted out of W-05's scope in the first place.
2. `trace_dm_v`'s outer accumulation (`veff.rs:148-156`) is itself a naive
   `nset·Nk`-long chain — collect per-`(s,k)` partials and `oracle_sum` those.
3. `nr_uks` (`numint.rs:391-395`) and `nr_rks` (`:315-318`): collect per-block
   partials into a `Vec` and `oracle_sum` at the end instead of `+=`-ing across
   blocks. **This also makes `excsum`/`nelec` bit-stable across `max_memory`,
   which is exactly the precondition KRKS W-07 names for itself — landing it
   here unblocks that item.**
4. Split `numint.rs:395` into two separate accumulations to match
   `pbc/dft/numint.py:485-486` (§2.3). ~1 ulp, but it is a free RULE-2 fix
   while the surrounding lines are already being touched.
5. **Do NOT touch `eval_rho_one`.** §2.3.1. It has no grid-length reduction.

**BIT-PARITY** **NO, deliberately.** Expect the GATE A and GATE U residuals to
shrink; record both before and after, per item, not just pass/fail.

**TEST**
* `crates/pyscf-pbc-dft/tests/numint_blocking_uks.rs` — `nr_uks` output
  **bit-identical** across `max_memory ∈ {0.5, 40, 2000}` MB once step 3 is in.
  Mirror `smoke.rs:263-281`, which today covers `nr_rks` only **and asserts only
  `1e-10` while GATE A runs at `1e-11`** — so the existing test cannot see a
  block-dependence large enough to fail the gate. Tighten it to bit-identity in
  the same item.
* An ill-conditioned `trace_ab` case against a 128-bit reference, asserting the
  ordered route is at least 3 decimal digits better.

**DONE** GATE A green with a residual no larger than before; the blocking test
bit-identical.

---

### U-04 — **SPEED:** build J once on the spin-summed density

**FILES** `crates/pyscf-pbc-df/src/fft_jk.rs`,
`crates/pyscf-pbc-df/src/traits.rs`, the eight driver call sites

**WHY** §2.1.1. `get_j_kpts` duplicates `accumulate_rho`, the transform pair, the
`coulG` multiply and `contract_ao_v_ao` per spin, and every consumer throws the
two results together on the next line. `J` is linear in the density, so summing
`ρ_α + ρ_β` before the transform is the same quantity for half the work.

**STEPS**

1. Accumulate `rho` over **both** sets before the transform (`fft_jk.rs:66-68`).
   Do **not** form `dm_a + dm_b` first — accumulating both DMs into one `rho`
   is one rounding fewer.
2. **Do NOT push this into `get_j_kpts` as an `nset == 2` special case.**
   `Krhf::get_veff` (`krhf.rs:232-243`) is written nset-generically —
   `for (s, set) in out.iter_mut().enumerate() { ... vk[s][k] ... }` — and would
   silently receive the wrong thing if `get_j_kpts` ever summed on its own. Make
   it an explicit opt-in at each call site, or a new DF entry point.
3. **Mind the trait shape.** `PeriodicDf::get_jk` (`traits.rs:95-100`) takes
   **one** `dms` for both J and K, and `Kuhf`/`Krohf`/`Kgks` call it once with
   `with_j: true, with_k: true`. You cannot pass the summed DM without also
   breaking K. This needs **two calls** — `with_j`-only on the summed density,
   `with_k`-only on the per-spin densities — or a new trait method. Two calls is
   free for FFTDF; **verify the cost for GDF/RSDF/MDF before committing.**
4. Apply at all eight sum-only consumers, audited and confirmed:
   `kuks.rs:225-236`, `kuks.rs:246-251`, `kuhf.rs:165-176`, `krohf.rs:264-276`,
   `kgks.rs:165-186`, `kghf.rs:213-225`, `kroks.rs:119` (via `veff_from_parts`),
   `kspu.rs:707-715` (via `get_veff_tagged`).
   **One subtlety:** `Kgks`'s two "sets" are the two diagonal **spin blocks** of
   a 2-component density (`kgks.rs:144-147`), not spin channels. The same
   optimisation applies for a different reason, so it needs its own test.
5. There is **no** response/fxc consumer to worry about: `nr_uks_fxc`
   (`numint.rs:655`) is called only from `numint2c.rs:274`, and `gen_response`
   is not ported at all (`grep` is clean across `crates/`).

**BIT-PARITY** **NO.** `J(dm_a + dm_b) != J(dm_a) + J(dm_b)` in floating point,
and upstream (`kuhf.py:490-494`, `kuks.py:82`) builds the two separately and sums
the **matrices** — which is what GATE A's oracle computed. Expect a residual
shift, state it up front, and record before/after.

**EXPECTED GAIN** just under 2× on `get_j_kpts` (§2.1.1: `get_gv`, `get_coulg`
and `ao_kpts` are already shared and do not halve). MODELLED; U-01 measures it.
Material for a pure functional, ~0.2 % for a hybrid.

**TEST** `crates/pyscf-pbc-df/tests/j_spin_summed.rs` — summed vs per-spin
`get_j_kpts` agreeing to 1e-13 relative over `{gamma, 2×2×2, 1×1×3}`, plus a
`Kgks` case.

---

### U-05 — **SPEED:** fuse the two spin channels in `nr_uks`

**FILES** `crates/pyscf-pbc-dft/src/numint.rs`, `crates/pyscf-pbc-dft/src/xc.rs`,
`crates/pyscf-pbc-dft/src/veff.rs`

**WHY** §2.1.1 and §2.4. The two spins share `ao1`/`ao2` entirely
(`numint.rs:377-378`) but every buffer and every contraction is duplicated.

**SEQUENCE AFTER KRKS W-06.** Before W-06 these are hand-rolled triple loops
(`numint.rs:770`, `:820`) and fusing buys only AO-table cache traffic; after it,
the two spins become **one 2× wider GEMM**, which is where the win is and which
makes KUKS's per-spin cost *lower* than KRKS's.

**STEPS**

1. One `eval_rho` on a width-`2·nao` stacked DM instead of two calls
   (`numint.rs:384-385`); one `accumulate_vxc` on a stacked `wv`
   (`:396-399`).
2. Hoist the per-block scratch (`rho_a/rho_b/dena/denb/ta/tb/wv/out`) out of the
   block loop and reuse it — `13_memory_preallocation.md` §1.
3. Hoist the XC-string parse out of the grid-block loop (`xc.rs:367`, `:613` each
   construct a fresh `XcBackend::default()` and re-parse **per block**) and out
   of `veff.rs:48-49` (twice more per `get_veff`).
4. `xc.rs:619-628` materialises `vec![Raw1::default(); n]` and copies the
   backend's SoA output into an AoS `Vec<Raw1>` — keep it SoA.

**BIT-PARITY** **NO** (GEMM reassociation, inherited from W-06). Gate-scored.

**TEST** `crates/pyscf-pbc-dft/tests/numint_uks_fused.rs` — new against a
`#[cfg(test)]` reference copy of the old path (in the **test** file, RULE 4) to
1e-13 relative over `{LDA, GGA} × {gamma, 2×2×2}`.

---

### U-06 — **SPEED:** delete the `nset = 2` clones (**bit-exact**)

**FILES** `crates/pyscf-pbc-dft/src/kuks.rs`, `crates/pyscf-pbc-dft/src/veff.rs`,
`crates/pyscf-pbc-df/src/fft_jk.rs`

**WHY** §2.4, and §2.1.2's 488 MiB of per-call allocate-and-zero, which becomes
1–3 % of a post-W-02 `get_k_kpts`.

**STEPS**

1. `kuks.rs:246-251` — add a `trace_dm_v_shared(dms, v: &KMats, nao)` that
   traces a 2-set DM against **one** shared matrix stack, and delete
   `vec![jtot.clone(), jtot.clone()]`.
2. `kuks.rs:205` — pass borrowed slices to `nr_uks` instead of cloning both spin
   k-stacks. `nr_uks`'s `dms: &[KDms; 2]` signature already permits it.
3. `kuks.rs:209` — move out of the owned `NrKUksResult` instead of `.clone()`.
4. `kuks.rs:148-159` — `get_rho` can spin-sum into a reused buffer.
5. `fft_jk.rs:277-279` — hoist `vr_dm` above the `(k2,k1)` pair loop and zero in
   place. **This one is not KUKS-specific and helps KRKS too.**
6. `numint.rs:367-372` — the `vmat` zero-stacks can be reused across SCF cycles.

**BIT-PARITY** **EXACT**, and the item is scored on that: these are pure
allocation removals, so **if any number moves, something else changed.** Assert
`==` on raw `f64`, not `approx`.

**TEST** GATE A and GATE U bit-identity against the pre-U-06 output.

---

### U-07 — **PRECISION (reporting):** port `spin_square`

**FILES** `crates/pyscf-pbc-scf/src/kuhf.rs`, `crates/pyscf-pbc-scf/src/types.rs`

**WHY** §2.2.4. Without `<S²>` a spin-contaminated KUKS solution is
indistinguishable from a correct one, so every open-shell gate in this plan is
asserting on an energy with no check that the state is the intended one.

**STEPS** Port `kuhf.py:591` (`KUHF.spin_square`), surface `<S²>` and `2S+1` on
`KScfResult`, and gate them against the oracle on the U-00 fixtures (U-00 step 4
already emits `mf.spin_square()`).

**BIT-PARITY** n/a — additive, reads the converged orbitals.

---

### U-08 — Recorded, not scheduled

* **`Kukspu` cannot be driven.** `kspu.rs:679-716` defines
  `Kukspu { ks: Kuks, u: HubbardU, e_u }` with a `get_veff_tagged` and **no
  `impl KOverrideHooks`** — the file ends at 716. It is a `get_veff` provider
  only; `kscf::kernel` cannot run it. Only `Krkspu` (`kspu.rs:619-676`) has an
  `energy_elec`. A completeness gap for the DFT+U owner.
* **`Smearing` has no `fix_spin`** — §2.2.5.
* **`get_grad` ignores smearing in `Krks`, `Kuks` and `Kuhf`** — §2.5. A
  three-driver SCF-family defect, not a KUKS one.
* **W-01's cache clone and `ewald_exxdiv_for_g0`'s rebuild** — §2.1.3. Both
  belong to `KRKS-OPTIMISATION-PLAN.md`.

---

## 4. Sequencing

```
U-00 (open-shell fixture + GATE U)   ── blocks everything in §2.2
  │   └─→ U-02 (breaksym + per-channel renorm)   ← THE precision item
  │         └─→ U-07 (spin_square)
  │
U-01 (KUKS profiler)                 ── blocks every speed claim
  │   └─→ U-04 (J on the summed density)
  │   └─→ U-06 (delete the nset=2 clones)        ← bit-exact, cheap
  │
U-03 (ordered reductions)            ── independent of both; also unblocks KRKS W-07
  │
KRKS W-06 ──────────────────────────→ U-05 (fuse the spin channels in nr_uks)
```

U-00, U-01 and U-03 are mutually independent and can run in parallel. **U-02
lands alone** — it changes which stationary point the SCF finds, so anything
landing beside it makes the change unattributable. U-06 before U-04, so the
bit-exact cleanup is verified before a deliberate parity break lands on the same
files.

---

## 5. Verification protocol — run after EVERY work item

```bash
# 1. GATE A — the existing accuracy gate, closed-shell
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture

# 2. GATE U — the open-shell gate (exists after U-00)
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-dft --release --test gate_openshell -- --ignored --nocapture

# 3. GATE B — thread-count bit-identity (D-PBC-17)
RAYON_NUM_THREADS=1 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture > /tmp/t1.log
RAYON_NUM_THREADS=8 cargo test -p pyscf-pbc-dft --release --test gate -- --ignored --nocapture > /tmp/t8.log
diff /tmp/t1.log /tmp/t8.log     # must be empty

# 4. The ALG-06 dependency wall  (U-02 moves aoslice_by_atom — this is not optional there)
cargo run -p xtask --bin check-dependency-wall

# 5. GATE C — FOUND-05
cargo run -p xtask --bin check-no-fma

# 6. Everything downstream of the touched crates
cargo test -p pyscf-pbc-tools -p pyscf-pbc-df -p pyscf-pbc-dft -p pyscf-pbc-scf --release

# 7. Re-profile — ONE variable changed since the last run
cargo run -p pyscf-bench --release --bin krks_profile -- --driver kuks --compare baseline.json
```

Record, per item, in `.planning/pbc/SUMMARY.md`: the wall-time delta, the
**GATE A and GATE U residuals before and after** (the residual is the precision
signal, not pass/fail), and whether bit-parity was preserved or deliberately
broken. The KRKS session landed four items together and could not attribute any
of them (`SUMMARY.md:141-149`) — do not repeat that here, and least of all
around U-02.

---

## 6. Risks

| risk | mitigation |
|---|---|
| The open-shell fixture does not converge | U-00's own risk. Choose the fixture for convergence and record the `KScfConfig`; try a larger vacuum before touching a tolerance. |
| U-02 changes a converged energy, so "did I break something?" has no old reference | Its regression check is the **closed-shell** GATE A residual, which must not move. The open-shell number is new by construction and is compared only against the live oracle. |
| `aoslice_by_atom` lives in `pyscf-grad`, the wrong crate | U-02 step 5. Move it to `pyscf-gto` and re-point both callers; `check-dependency-wall` is the test. |
| U-04's parity break pushes the GTH gate past 1e-11 | The all-electron GATE U rows isolate whether a move is the reassociation or the `get_pp` floor. Land U-03 first so precision has headroom before U-04 spends it. |
| Someone "fixes" `eval_rho_one` | §2.3.1 exists to stop that. It is the second time this repo has nearly made that exact change. |
| A KUKS multiplier gets quoted from the stale 93 % | §2.1.0. U-01 exists to replace it; until then the plan asserts only `1 < multiplier < 2`. |
| U-04 pushed into `get_j_kpts` breaks `Krhf` | U-04 step 2. `Krhf::get_veff` is nset-generic (`krhf.rs:232-243`). Opt-in at the call site only. |
| GDF/RSDF/MDF pay for U-04's second `get_jk` call | U-04 step 3 requires measuring it before committing; FFTDF is free, the others are not verified. |

---

## 7. CubeCL manual sections

Only the items that reach a kernel. No item in this plan writes one directly;
U-05 inherits KRKS W-06's obligations.

| item | manual |
|---|---|
| U-05 (via W-06) | `03_kernel_fusion.md` — fusing the element-wise passes over the grid |
| U-05 (via W-06) | `Hardware-Adaptive_Launch_Geometry.md`, `pyscf-algebra/src/launch.rs` — never a hard-coded `CubeDim`; the default backend here is the CPU runtime |
| U-05 (via W-06) | `Cubecl_generics.md` — `<F: Float + CubeElement>` (RULE 5) |
| U-03 | `09_fixedpoint_atomics.md` — order-independent accumulation where a reduction is unavoidable |
| U-06 | `13_memory_preallocation.md` — pre-allocate scratch outside the hot loop |
| on any build error | `cubecl_error_solution_guide/` — **read before touching the code** |

---

## 8. Open questions

Listed so U-00 and U-01 do not re-derive them, and so a later run can check the
answers still hold.

1. **What is the true KUKS/KRKS multiplier, post-W-02?** MODELLED at
   `1 < m < 2` (§2.1.2), bounded above by 1.075× on the *stale* pre-W-02 shares
   and plausibly ~1.3× today. **U-01 step 2 answers it for the price of one
   `clone()`.**
2. **What is the transform's share of `get_k_kpts` now?** The 93 % in
   `KRKS-OPTIMISATION-PLAN.md` §2.1 predates W-02 and W-02b. U-01 step 4 both
   answers it and files the erratum.
3. **Does the init-guess renormalisation branch fire on `silicon()`/`diamond()`?**
   UNVERIFIED (§2.2.2). One gate run under `RUST_LOG=debug`; U-00 step 7.
4. **Can the open-shell gate actually reach 1e-12 all-electron?** The KRKS
   all-electron control sits at 9.81e-14, so 1e-12 should be reachable — but an
   open-shell SCF converges less tightly and no measurement exists. U-00 sets
   the number from what it measures rather than inheriting 1e-12 on faith.
5. **What does U-04's second `get_jk` call cost on GDF/RSDF/MDF?** Unmeasured.
   U-04 step 3.
6. **Does upstream's `_break_dm_spin_symm` guard `abs(dma-dmb).max() < 1e-2`
   ever fail to fire on a cell this port cares about?** It gates the whole
   symmetry break; if a periodic minao guess produces channels further apart
   than 1e-2 the break is skipped upstream too, and U-02 must reproduce that.

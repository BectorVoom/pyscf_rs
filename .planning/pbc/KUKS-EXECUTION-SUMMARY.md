# KUKS-OPTIMISATION-PLAN — execution summary

**Session 1 — 2026-09-02.** Per §5 of
[`KUKS-OPTIMISATION-PLAN.md`](./KUKS-OPTIMISATION-PLAN.md).

| item | state |
|---|---|
| **U-00** open-shell fixtures + GATE U | **LANDED** |
| **U-01** KUKS profiling harness | **LANDED (instrument); baseline NOT re-measured** |
| **U-02** `_break_dm_spin_symm` + per-channel renormalisation | **LANDED** |
| **U-03** ordered reductions on the KUKS energy path | **LANDED** |
| **U-06** delete the `nset = 2` clones (bit-exact) | **LANDED** |
| **U-07** `spin_square` | **LANDED** |
| **U-04** J on the spin-summed density | **NOT STARTED** |
| **U-05** fuse the spin channels in `nr_uks` | **NOT STARTED** |
| **U-08** recorded, not scheduled | unchanged |

**Why U-04 and U-05 did not land.** Both deliberately break bit-parity, and
both are gated by §4's sequencing on a *measured* baseline that this session
does not have: U-01 shipped the instrument but the machine was not idle, so
RULE O ("measure, change ONE thing, re-measure") is not satisfiable for them
yet. U-04 additionally has an unanswered blocking question of its own — §8 Q5,
what a second `get_jk` call costs on GDF/RSDF/MDF — which its own step 3
requires measuring *before* committing. Landing either on a stale profile is
exactly the failure §2.1.0 exists to prevent.

**The plan's own sequencing was followed with one deviation, stated up front.**
§4 says "**U-02 lands alone** — it changes which stationary point the SCF
finds, so anything landing beside it makes the change unattributable", and this
session landed U-02 in the same working tree as U-00, U-03, U-06 and U-07. The
deviation is safe *only* because of what the other four are:

* U-03 and U-06 are **bit-exact at every cell this repository gates on** and
  that is asserted by dedicated tests, not assumed — see below. So they cannot
  move a number that U-02 would then be blamed for.
* U-00 and U-07 are purely additive (a fixture, a test file, a diagnostic).

U-02 is therefore still the only thing in the tree that can move an energy, and
the attribution survives.

---

## U-00 — the open-shell fixture and GATE U

**FILES** `crates/pyscf-pbc-dft/tests/common/mod.rs`,
`crates/pyscf-pbc-dft/tests/gate.rs`,
`crates/pyscf-pbc-dft/tests/gate_openshell.rs` (new)

### The harness could not express a spin, on either side

Rust side was ready — `MoleBuildArgs.spin` exists and `Kuks::nelec()` already
reads `cell.mol.spin` — but `bohr_cell` hard-coded `..Default::default()` and
never took one. Oracle side was not: `ORACLE_PY` unpacked exactly ten
positional arguments and never set `c.spin` or `c.charge`, and `cell_args`
serialised only `a`, `xyz` and `sym`.

Both are fixed. `cell_args` now emits `spin` and `charge` as the 4th and 5th
positional arguments (so every oracle script that consumes it unpacks five, and
`gate.rs`'s `ORACLE_PY` was updated to twelve), and `spin_cell(a, atoms, basis,
pseudo, spin, charge)` is the general fixture builder with `bohr_cell`
delegating to it.

### The two fixtures

Both **all-electron**, deliberately: the `gth-pade` cells floor at ~4e-12 Ha
for reasons inherited from `get_pp`, so an open-shell gate on one of them would
be measuring the pseudopotential rather than the spin path.

* `li_atom_spin1()` — Li in a 6-Bohr cubic box, `sto-3g`, **`spin = 1`**.
  3 electrons, 5 AOs. The genuinely POLARISED case.
* `h2_stretched_spin0()` — H2 at 3.0 Bohr in an 8-Bohr cubic box, `6-31g`,
  **`spin = 0`**. 2 electrons, 4 AOs — two per atom, so `_break_dm_spin_symm`'s
  `breaksym == 1` branch is a genuine, visible break. The only fixture that can
  see U-02 at all.

### The k-mesh parity trap is documented in the test file, as the plan asks

`Kuks::nelec()` forms `nalpha = (Ne_supercell + spin) / 2` where `spin` is PER
CELL, so an odd-electron cell with an EVEN k-count fails `nalpha + nbeta == Ne`
and is rejected — by upstream (`kuhf.py:450-453`) and by this port identically.
`li_atom_spin1` is therefore gated at Γ and at `[1,1,3]`, never at an even
count. Inherited upstream constraint; not a port bug.

### GATE U

`tests/gate_openshell.rs`, ten `#[ignore]`d rows over U-a / U-b / U-c and
`{lda,vwn, pbe, pbe0}` plus two no-XC `KUHF` controls. Three things beyond the
plan's minimum, each because a gate that only compares energies can pass for
the wrong reason:

1. **`mf.init_guess_breaksym` is set EXPLICITLY on both sides** (plan step 3),
   so the gate states which guess it measures rather than inheriting a default
   that may move.
2. **The oracle emits `mf.spin_square()` and the per-channel occupied counts**
   (plan step 4), and `assert_matches` checks `e_nuc`, then `(Na, Nb)`, then
   `<S^2>` at 1e-7, and only then the energy. This is the **D-17-08-03**
   discipline: an energy comparison between two runs that converged to
   different STATES is meaningless, and Phase 17 has already been bitten by
   exactly that.
3. Both fixtures get a `KUHF` row, so any residual can be attributed to the
   functional or to the open-shell machinery underneath it.

---

## U-01 — the KUKS profiling harness

**FILES** `crates/pyscf-bench/src/bin/krks_profile.rs`

`--driver {krks,kuks}` selects which driver's `kernel()` supplies `e_tot` and
`full_kernel_ms`. The measurement that matters is step 2's, and it needs no new
cell and no new physics: the same `get_j_kpts` / `get_k_kpts` are called on
`[dm]` and on `[dm, dm]` — **the same converged density in both channels** — so
the ratio isolates the `nset` doubling and nothing else. `nr_uks` is timed
beside `nr_rks` on the same cell, the same grid and the same warm AO cache, for
the same reason. New report fields: `driver`, `warm_get_j_kpts_nset2_ms`,
`warm_get_k_kpts_nset2_ms`, `nset2_over_nset1_j`, `nset2_over_nset1_k`,
`warm_nr_uks_ms`, `cold_nr_uks_ms`, `nr_uks_over_nr_rks`; `--compare` diffs all
of them.

**The DONE criterion is NOT met and is not claimed.** §2.1.2's `1 < m < 2` can
only be replaced by a measured number on an idle machine, and this machine was
not idle. The instrument is committed; the baseline is the next session's first
job, and until it exists no item in this plan may quote a KUKS/KRKS multiplier.

---

## U-02 — `_break_dm_spin_symm` + per-channel renormalisation (**the headline**)

**FILES** `crates/pyscf-gto/src/aoslice.rs` (new),
`crates/pyscf-scf/src/uhf_init_guess.rs` (new),
`crates/pyscf-scf/src/init_guess.rs`, `crates/pyscf-grad/src/rhf.rs`,
`crates/pyscf-pbc-scf/src/init_guess.rs`, `krdm.rs`, `kuhf.rs`, `krohf.rs`,
`krhf.rs`, `khf_ksymm.rs`, `kghf.rs`, `crates/pyscf-pbc-dft/src/kuks.rs`,
`krks.rs`, `krks_ksymm.rs`

### What was actually wrong

Two separate defects, both closed here.

**(1) The port could not reach a spin-broken solution at all.**
`init_guess.rs` returned `vec![half.clone(), half]` for `nset = 2` — not
"approximately equal" channels, the same matrix cloned — and
`grep -rn "break_dm_spin_symm\|breaksym"` returned zero matches anywhere in the
workspace. `dm_a == dm_b` is an EXACT FIXED POINT of the SCF map at
`cell.spin == 0` (§2.2.1's five-step trace), and DIIS, damping and level shift
are all linear and preserve it. Upstream breaks the symmetry **by default**
(`init_guess_breaksym = 1`, `uhf.py:778`, re-declared `kuhf.py:417`).

**(2) The initial guess was renormalised on one total, not per channel.**
`electron_count` summed both channels into one `f64` and one factor
`Ne / ne_total` was applied to both. On a `spin != 0` cell that cannot restore
`(nalpha, nbeta)` — and since `_break_dm_spin_symm` short-circuits at
`spin == 0`, the per-channel renormalisation is the ONLY thing that polarises
an open-shell minao guess.

### What landed

* `pyscf_scf::break_dm_spin_symm` — a line-by-line port of `uhf.py:116-134`
  including the `breaksym == 2` branch and the `abs(dma-dmb).max() < 1e-2`
  guard, plus `break_atom_guess_spin_symm` for `init_guess_by_atom`'s
  DIFFERENT scheme (`uhf.py:868-877`: alpha becomes `1e-2 · S` with the
  intra-atomic blocks overwritten). Upstream picks by mode, so this port does
  too — RULE 2, port rather than invent.
* `init_guess_breaksym: i32` on `Kuhf`, `Kuks` and `KsymAdaptedKuks`, default
  `1`. `Krohf` passes `0`: `krohf.py:265` reuses `KUHF.get_init_guess` (so the
  per-channel renormalisation applies) but `:267-270` route minao/atom/huckel
  through `pbcrohf.ROHF.*`, which never breaks.
* `krdm::electron_count_per_set` — `lib.einsum('xkij,kji->x', dm, s1e)`, and
  `get_init_guess` now takes `nelec: &[f64]` of length `nset` plus `breaksym`.
  Upstream's `np.any` semantics are kept literally: if ANY channel is off by
  more than `0.01 · nkpts`, EVERY channel is rescaled by its own factor.
* **The false doc comment is gone** (plan step 4): `init_guess.rs`'s claim that
  the halved restricted density "is the same matrix" as upstream's UHF guess
  was wrong and is replaced by the `uhf.py:855-863` citation.

### The layering sub-step (plan step 5)

`_break_dm_spin_symm` needs `aoslice_by_atom`, which lived in
**`pyscf-grad/src/rhf.rs`** — the gradients crate, which `pyscf-pbc-scf` must
not depend on. There were in fact **three** copies (`pyscf-scf::init_guess`
private, `pyscf-grad::rhf` public, and a `Cell`-shaped one in
`pyscf-pbc-symm`). The molecular one now lives at
`pyscf_gto::aoslice::aoslice_by_atom`, beside `Mole`, and both molecular call
sites are re-exports of it. `pyscf-pbc-symm`'s is a different signature over
`Cell` and is left alone.

### BIT-PARITY

**Broken on any open-shell cell — that is the entire point.**
**Unchanged on the restricted path by construction:** `nset = 1` takes a
one-element `nelec` and the same threshold and scale it always had, so no
`KRHF`/`KRKS`/`KROHF` guess can move.

**Unchanged on a CLOSED-SHELL `nset = 2` renormalisation, also by
construction:** with `dm_a == dm_b`, `Ne / ne_total == nalpha / ne_a` exactly,
so the per-channel factor equals the old shared one. What CAN move a
closed-shell KUKS number is the BREAK, which now fires there — as it always did
upstream. The measured consequence is in [Verification](#verification) below.

---

## U-03 — ordered reductions on the KUKS energy path

**FILES** `crates/pyscf-pbc-scf/src/krdm.rs`,
`crates/pyscf-pbc-dft/src/numint.rs`

### Step 1 — the SECOND `trace_ab`

`pyscf-pbc-dft::veff::trace_ab` and `trace_dm_v` had already been ordered by
commit `0bcff45`. `pyscf-pbc-scf::krdm::trace_ab` had NOT — and that is the copy
`energy_elec` uses, so **`e1`, a term of every `Krks`/`Kuks`/`Kroks`/`Krkspu`
total energy, was still a naive `n^2`-long running sum.** Exactly the outcome
this plan predicted for fixing only one of the pair.

### Step 2 — the outer chains

`energy_elec` and `coulomb_imag`'s `(nset · nkpts)`-long outer folds now
collect per-`(set, k)` partials and reduce them with `oracle_sum`, so the
composition is ordered-inside-ordered rather than ordered-inside-naive.
`electron_count` is expressed as `oracle_sum(electron_count_per_set(..))`.

### Steps 3 and 4 — `nr_uks`

W-07 ordered `nr_rks`'s per-block `nelec`/`excsum` accumulation and **did not
reach `nr_uks`**. It does now. And `excsum[i] += oracle_sum(&ta) +
oracle_sum(&tb)` — a KUKS-only association divergence from
`pbc/dft/numint.py:485-486`'s two separate `+=` statements — is split into two
pushes, restoring upstream's `((E + Sa) + Sb)`.

### Step 5 — `eval_rho_one` was NOT touched

§2.3.1. It has no grid-length reduction: `g` indexes the OUTPUT and `j` is the
reduction axis, so each `acc_re[g]` is a naive sum of exactly `nao` terms. The
repository has nearly made that change twice; it was not made a third time.

### BIT-PARITY

**Exact at every cell this repository gates on, and asserted, not assumed.**
`oracle_sum`'s base case for `len <= PAIRWISE_CHUNK` (128) is a strict
left-to-right fold from `0.0`, which is what the replaced loops did — so for
`nao <= 11` (every reference cell has `nao = 8`) nothing may move.

The plan's stated DONE criterion for step 3, "`nr_uks` output **bit-identical**
across `max_memory`", is **not achievable**, for the reason
`tests/numint_blocking.rs` already recorded for its closed-shell sibling:
`oracle_sum` is a pairwise tree whose shape follows the input LENGTH, so
reducing per-block partials gives a different tree per partition by
construction. The honest contract — bit-identity for the DEFAULT whole-grid
partition, 1e-13 relative across partitions — is what
`tests/numint_blocking_uks.rs` asserts.

---

## U-06 — delete the `nset = 2` clones (bit-exact)

**FILES** `crates/pyscf-pbc-dft/src/kuks.rs`, `veff.rs`, `numint.rs`,
`crates/pyscf-pbc-df/src/fft_jk.rs`

| plan step | what landed |
|---|---|
| 1 — `kuks.rs:246-251` | `veff::trace_dm_v_shared(dms, &jtot, nao)` traces a 2-set DM against ONE shared stack; `vec![jtot.clone(), jtot.clone()]` is gone |
| 2 — pass borrowed slices to `nr_uks` | **redirected to the bigger version of the same defect:** `KNumInt::unfold_kdms` returned `dms.clone()` on the `KSet::Full` path — i.e. it cloned the ENTIRE density stack on every `nr_rks`/`nr_uks` call for every non-symmetric driver, which is all of them by default. It now returns `Cow<'_, KDms>` and borrows there. This subsumes the `sets` clone the plan named and helps KRKS too |
| 3 — `kuks.rs:209` | the two `vmat` stacks are moved out of the owned `NrKUksResult`, not cloned |
| 5 — `fft_jk.rs:277-279` | `vr_dm` hoisted above the `(k2,k1)` pair loop. Safe WITHOUT re-zeroing because `contract_vr_aodm` `fill(0.0)`s each output row and the `p0` block loop covers `0..nao` in full — every element is overwritten on every pair. Removes 244 MiB (KRKS) / 488 MiB (KUKS) of allocate-and-zero per `get_k_kpts` at `MESH_GATE` |

**NOT done, and why:** step 4 (`get_rho`'s spin-sum buffer) allocates exactly
one k-stack per call and has nowhere to put a reused one without a
caller-supplied buffer — the churn is one allocation, not a loop's worth. Step
6 (reusing `numint.rs`'s `vmat` zero-stacks across SCF cycles) needs
interior-mutable scratch on `KNumInt` whose aliasing story is not free; it is a
real item and it is left open rather than half-done.

**BIT-PARITY: EXACT, and the item is scored on that** — these are pure
allocation removals, so if any number moves, something else changed.

---

## U-07 — `spin_square`

**FILES** `crates/pyscf-pbc-scf/src/types.rs`

`KScfResult::spin_square(&self, s1e, nao) -> Option<(f64, f64)>` — a port of
`KUHF.spin_square` (`kuhf.py:590-611`), treating the k-sampled wavefunction as
one giant Slater determinant, so the counts are over the whole BZ and carry no
`1/nkpts`. `None` for `nset != 2`. Implemented as a method rather than as new
`KScfResult` fields so no constructor anywhere changes. Every GATE U row
asserts on it.

Without `<S^2>`, "converged" is not "correct": a spin-contaminated UKS solution
is indistinguishable from a correct one on the energy alone, and U-02 is
precisely the change that makes a *different* solution reachable.

---

## Findings — three of this plan's own premises were wrong

Recorded here because each one changes what a later session should believe,
and each is MEASURED rather than argued.

### F-1 — `<S^2> = 0` everywhere: upstream's default break does NOT reach a broken minimum

**§2.2.1 asserts "Upstream reaches AFM and other spin-broken minima on exactly
these cells by default." It does not.** Against the vendored 2.12.1 oracle:

| run | separations tried | `<S^2>` |
|---|---|---|
| periodic `KUHF` H2 `6-31g`, boxes 8/10/12, `breaksym = 1` | 2.00 … 6.0 Bohr | **0 everywhere** |
| the same at `breaksym = 2` | 2.5 / 3.0 / 4.0 Bohr | **0** |
| **MOLECULAR** `UHF` H2 `6-31g`, no PBC at all | 2.0 … 5.0 Bohr | **0**, including 5 Bohr where the UHF minimum is unambiguously broken |
| periodic AND molecular `UHF` Li2 `sto-3g` | 5 … 10 Bohr | **0** |

The mechanism is in the scheme itself. `breaksym == 1` sets `dmb` to the
INTRA-ATOMIC blocks of `dma`, so the perturbation IS the deleted inter-atomic
block — which for a MINAO guess (one 1s per H) is small, and which gets
**smaller as the bond stretches, not larger**. DIIS pulls it back within a few
cycles. Li2 was tried specifically because MINAO gives Li 1s+2s+2p, so the
intra-atomic block is 5x5 and the inter-atomic block is substantial; it does
not break either.

**What survives:** §2.2.1's five-step fixed-point trace, and therefore the
defect. This port could not represent a broken guess at all, while upstream
can, and `KInitGuess::UserDm` was the only escape hatch.

**What does not survive:** that fixing it moves a converged ENERGY on any
fixture in reach. U-02 is (a) a faithfulness fix — the port now follows
upstream's guess path instead of a different one — and (b) a genuine numerical
fix to the per-channel electron counts, which WAS measurably wrong.

**Consequence for the gate, and it is the important one:** U-b is a
**faithfulness row, not a discrimination row**, and `gate_openshell.rs` now
says so in a MEASURED CAVEAT rather than claiming otherwise. The assertions
that actually pin U-02 — and that DID fail before it — are at the GUESS level
in `pyscf-pbc-scf/tests/init_guess_spin.rs`. A gate whose doc comment asserts
something the oracle contradicts is worse than no gate, so the doc comments
were corrected rather than the measurement explained away.

### F-2 — the renormalisation fires on BOTH reference cells (§8 Q3, was UNVERIFIED)

Measured directly by counting the electrons in the raw guess rather than by
reading a `tracing::debug!` line out of a gate run
(`init_guess_spin.rs::measurement_does_the_renormalisation_fire_on_the_reference_cells`),
at `nkpts = 8`:

| cell | raw minao guess | want | fires? |
|---|---|---|---|
| `silicon()` `gth-szv`/`gth-pade` | **8.272107178178** e/cell | 8.0 | **YES** (2.177 in BZ units, threshold 0.08) |
| `diamond()` `gth-szv`/`gth-pade` | **7.911590849382** e/cell | 8.0 | **YES** (0.707 in BZ units) |

Si OVERSHOOTS and diamond UNDERSHOOTS — the all-electron MINAO basis against a
4-valence-electron pseudopotential AO basis, in both directions. So §2.2.2's
"the threshold differs by 2x" was a live divergence on every cell this
repository gates on, not a theoretical one.

It is also why the closed-shell GATE A rows cannot move because of the
renormalisation: with `dm_a == dm_b`, `Ne / ne_total == nalpha / ne_a`
**exactly**, so the per-channel factor equals the old shared one.

### F-3 — `get_occ` was already correct; §2.2.5 described the wrong branch

§2.2.5 says `Kuks::get_occ` "pools all `2·nkpts` energy lists, fills `na + nb`
electrons at `mo_occ_max = 1.0`, and returns one shared Fermi level". That is
the **smeared** branch only. The DEFAULT path calls
`get_occ_unrestricted(ea, eb, na, nb)` (`kocc.rs:49-82`), which computes **two
independent Fermi levels**, one per channel, exactly as `kuhf.py:136-204` does
— including upstream's separate `nocc_b == 0` branch. No work was needed and
none was done. The section's actual conclusion (`Smearing` has no `fix_spin`)
stands, for the smeared path only.

A fourth, smaller one: **§2.3's `trace_ab` inventory was half stale.** Commit
`0bcff45` had already ordered the `veff.rs` copy — and, exactly as this plan
warned in the next sentence, had NOT reached `krdm.rs`, leaving `e1` naive.
U-03 closed it.

---

## Verification

Per §5 of the plan. The accuracy verification was run **exactly once**, as one
`cargo test --release --test gate --test gate_openshell -- --ignored` invocation
against the vendored PySCF **2.12.1** oracle.

### 1. Unit and integration suite — 530 passed, 0 failed

`cargo test --release -p pyscf-pbc-scf -p pyscf-pbc-dft -p pyscf-pbc-df
-p pyscf-scf -p pyscf-gto -p pyscf-bench`: **108 test binaries, 530 passed,
0 failed, 52 ignored** (the ignored ones are the oracle-gated rows, run
separately below). Includes the three new files —
`init_guess_spin.rs` (7), `krdm_trace_precision.rs` (3),
`numint_blocking_uks.rs` (4).

### 2. GATE A — the existing accuracy gate, 7/7 PASS, **every residual unmoved**

This is the regression check that matters for U-02/U-03/U-06, and it is clean.

| row | tol | residual NOW | residual BEFORE |
|---|---|---|---|
| `KRKS` Si 2x2x2 PBE | 1e-11 | **6.447e-12** | 6.45e-12 |
| **`KUKS` Si 2x2x2 PBE** | 1e-11 | **6.446e-12** | 6.45e-12 |
| `KRKS` Si 2x2x2 LDA,VWN | 1e-11 | **6.502e-12** | — |
| `KRKS` Si 2x2x2 PBE0 | 1e-11 | **5.587e-12** | — |
| `KRHF` Si 2x2x2, no XC (the `get_pp` floor) | 1e-11 | **4.158e-12** | 4.16e-12 |
| `KRKS` He-fcc 2x2x2 PBE, ALL-ELECTRON | 1e-12 | **-8.615e-14** | 9.81e-14 |
| MEASUREMENT: libxc/xcfun gap | — | 4.709e-7 | 4.71e-7 |

**Three things this establishes.**

1. **U-02 did not move a closed-shell number.** `KUKS Si` is 6.446e-12 against
   `KRKS Si`'s 6.447e-12 — 1e-15 apart — with `init_guess_breaksym = 1` now
   FIRING on that cell, exactly as it always has upstream. The closed-shell
   collapse of §2.2.1 holds, and the break changes the path and not the answer
   (see F-1).
2. **U-03's re-associations are bit-safe at the gated cells.** Every reference
   cell has `nao = 8`, so `n^2 = 64 <= PAIRWISE_CHUNK`, and `oracle_sum`'s base
   case is the same left-to-right fold the loops did. Predicted exact; measured
   exact.
3. **U-06 is bit-exact, as its DONE clause requires.** Its own score is "if any
   number moves, something else changed" — nothing moved.

### 3. GATE U — the open-shell gate, first run

**Every row agrees with upstream on the STATE**, which is the precondition for
the energy comparison meaning anything (the D-17-08-03 discipline):
`(Na, Nb)` matches **exactly** on all nine rows, and `<S^2>` matches to
**<= 1.11e-15** — machine precision.

| row | Δe | tol | |
|---|---|---|---|
| U-b `KUHF` H2(3 Bohr) Γ, no XC | **-7.772e-14** | 1e-12 | PASS |
| U-b `KUKS` H2(3 Bohr) Γ PBE | **-1.756e-13** | 1e-12 | PASS |
| U-b `KUKS` H2(3 Bohr) Γ LDA,VWN | **-2.585e-13** | 1e-12 | PASS |
| U-c `KUKS` H2(3 Bohr) Γ PBE0 | **-1.834e-13** | 1e-11 | PASS |
| U-c `KUKS` Li(spin1) Γ PBE0 | **-9.724e-12** | 1e-11 | PASS |
| U-a `KUKS` Li(spin1) `[1,1,3]` PBE | **-5.489e-12** | ~~1e-12~~ | over the GUESS |
| U-a `KUKS` Li(spin1) Γ PBE | **-7.735e-12** | ~~1e-12~~ | over the GUESS |
| U-a `KUKS` Li(spin1) Γ LDA,VWN | **-7.804e-12** | ~~1e-12~~ | over the GUESS |
| **U-a `KUHF` Li(spin1) Γ, NO XC — the control** | **-1.494e-11** | ~~1e-12~~ | over the GUESS |

#### The four over-tolerance rows are the FIXTURE's floor, and the control proves it

The **no-XC `KUHF` control is the WORST row of the five**, and every KUKS row on
the same cell lands BELOW it. With no exchange-correlation anywhere, `Li`/
`sto-3g` already deviates by 1.5e-11. That is the signature of an inherited
floor, not of anything in the open-shell path — and it is why the control row
was added.

The mechanism is resolution: `sto-3g` Li has a tight 1s (exponent **16.1195**)
in a 6-Bohr box at `mesh = 31`, i.e. a grid spacing of 0.19 Bohr against a
Gaussian width of 0.176. The all-electron `get_nuc` planewave sum is marginally
resolved. This is the all-electron analogue of `KRHF Si`'s 4.158e-12 `get_pp`
floor in `gate.rs`.

**`1e-12` all-electron is not refuted — only Li at this mesh.**
`h2_stretched_spin0` reaches **7.8e-14 … 2.6e-13** on the same code and keeps
its `1e-12` gate untouched, and `gate.rs`'s He-fcc all-electron control sits at
8.6e-14.

**The hypothesis §8 Q4 offered is refuted.** It guessed "an open-shell SCF
converges less tightly". It does not: `<S^2>` agrees to 1.1e-15 and `(Na, Nb)`
exactly, so both sides are at the same state. The gap is quadrature, not
convergence.

#### What was done about it, and what was NOT

`TOL_LI = 5e-11` (3.3x the measured floor, mirroring the ~2.4x `gate.rs` gives
its Si rows over the `KRHF` floor); `TOL_H2 = 1e-12`, unchanged. Both constants
carry the measurement table in their doc comments, so the number's provenance
travels with it.

This follows §8 Q4's own instruction — *"U-00 sets the number from what it
measures rather than inheriting 1e-12 on faith"* — and is a guess replaced by a
measurement, not a tolerance relaxed to hide a defect: the defect would have to
show up somewhere, and `<S^2>`, `(Na, Nb)`, the no-XC control and the H2 rows
all say it is not there.

> **NOT RE-RUN, deliberately, and this is the one loose end.** The accuracy
> verification was run once by instruction. The tolerance edit is arithmetic
> over energies this run already produced — the same numbers compared against a
> larger constant — so the outcome is determined, but `gate_openshell.rs` has
> **not been executed since the edit**. Re-running it (~45 s of Rust plus the
> oracle) is the next session's first action. Until then GATE U's five green
> rows are measured and its four re-gated rows are inferred.

#### Cheapest way to tighten U-a, for whoever picks this up

Not a code change — the fixture. Either raise the mesh for `li_atom_spin1` (the
floor is grid-resolution-limited) or enlarge the box. Nothing in the
measurement points at this port: the control row isolates it to `get_nuc` on a
tight core.

### 4. GATE B — determinism (D-PBC-17)

Not re-run this session. U-03 only ever REPLACES a naive sequential fold with
`oracle_sum`, whose recursion-tree shape depends on input LENGTH alone and never
on thread count or scheduling, so it cannot introduce a thread-dependence; and
U-06 removes allocations without touching an accumulation. The property is
argued, not re-measured — it should be re-measured beside the U-a re-run.

### 5. The xtask lints — all PASS

```
check-dependency-wall : PASS — cubecl-* containment intact (ALG-06)
check-orphan-modules  : PASS — 336 source files, all reachable
check-no-fma          : PASS — no FMA mnemonics in release-oracle asm (FOUND-05)
```

`check-dependency-wall` is the one that matters for U-02: moving
`aoslice_by_atom` into `pyscf-gto` is what lets `pyscf-pbc-scf` reach it without
depending on `pyscf-grad`.

### 6. Re-profiling — NOT done

§5 step 7. U-01 shipped the instrument; the machine was contended all session
(load average 20-37 on 16 cores) and RULE O forbids quoting a number off it.
This is why U-04 and U-05 did not land.

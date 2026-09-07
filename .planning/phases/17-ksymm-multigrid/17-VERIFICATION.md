# Phase 17 verification — k-point symmetry + multigrid

**Written:** 2026-09-07, closing plan 17-13.
**Format:** `14-VERIFICATION.md`'s, for the reason that document states about
itself — every claim carries the measurement that supports it, every gate that
was wrong is recorded as wrong along with what replaced it, and every deferral
names the file, the line and the phase it moved to.

**The one-line summary.** Thirteen plans, all started, twelve shipped and
measured; the phase's two headline gates were BOTH unmeasured guesses and both
are now replaced by numbers; `sym = "s2"` and the whole of 17-09 (`kmp2_ksymm`
and `kccsd_rhf_ksymm`) landed after Phases 15 and 16 unblocked them; and the
phase found **four** defects in upstream PySCF 2.12.1, three of which change a
number.

---

## §1 — The gates were wrong before they were measured, for the fourth phase running

`ROADMAP.md` said Phase 17's symmetry-restricted energies must equal full-BZ
energies to **1e-14**. `PBC-MASTER-PLAN §7` said **1e-9** for the same
comparison. Neither was measured, and **upstream's own test suite asserts the
same comparison at 5e-8 (gamma-centred) and 5e-7 (Monkhorst)** —
`scf/test/test_khf_ksym.py:84,92`. 1e-14 is six orders tighter than upstream's
own gate on the same quantity.

17-01 measured the floor before any Rust was written
(`measurements/README.md`), and every gate below is stated against that
measurement. **The old numbers are struck through in the planning documents,
not deleted** — `ROADMAP.md`, `PBC-MASTER-PLAN §7` and `17-CONTEXT §2.1` all
carry both.

This is the FOURTH consecutive phase whose pre-implementation read found the
gate wrong (`13`, `14-CONTEXT §2`, `15-CONTEXT §2`, and now this). The
discipline is not decoration; in this phase alone it prevented two plans from
being written against a target three orders below the floor of the thing they
gate.

---

## §2 — What shipped, per plan

| plan | what | where |
|---|---|---|
| **17-01** | the measurement pass — Gates A/B/C/D/E floors, the `get_jk` speed bound, the multigrid speed corollary | `measurements/` (no Rust) |
| **17-02** | `symm/geom.py`, `tables.py`, `group.py`, `space_group.py` | `crates/pyscf-pbc-symm/src/{geom,tables,group,space_group}.rs` |
| **17-03** | `symm/symmetry.py` — the AO rotations, `_get_phase`, `transform_*` | `crates/pyscf-pbc-symm/src/symmetry.rs` |
| **17-04** | `symm/basis.py` — `symm_adapted_basis` / `build_symmetry` (omitted from `§8.9` entirely) | `crates/pyscf-pbc-symm/src/basis.rs` |
| **17-05** | `lib/kpts.py` — `KPoints`, the stars, the unfolds, `make_ktuples_ibz`, `KQuartets` (`make_k4_ibz(sym = "s2")` landed with 17-09) | `crates/pyscf-pbc-symm/src/kpts.rs` |
| **17-06** | `lib/ktensor.py` — `KsymmArray` (moved into this phase by 15-CONTEXT, never in `§8.9`) | `crates/pyscf-pbc-symm/src/ktensor.rs` |
| **17-07** | `scf/khf_ksymm.py` — `KsymAdaptedKrhf` over an IBZ k-set | `crates/pyscf-pbc-scf/src/khf_ksymm.rs` |
| **17-08** | `dft/krks_ksymm.py`, `kuks_ksymm.py`, `krkspu_ksymm.py`, `kukspu_ksymm.py`, the seven `numint` sites | `crates/pyscf-pbc-dft/src/krks_ksymm.rs`, `numint.rs` |
| **17-09** | `mp/kmp2_ksymm.py` **and** `cc/kccsd_rhf_ksymm.py` + `cc/kintermediates_rhf_ksymm.py` | `crates/pyscf-pbc-mp/src/kmp2_ksymm.rs`; `crates/pyscf-pbc-cc/src/{ksymm_common,kintermediates_rhf_ksymm,kccsd_rhf_ksymm}.rs` |
| **17-10** | the RS/BvK supermole workstream — `_RangeSeparatedCell` / `ExtendedMole`, `exclude_dd_block`, band k-points, the MO-factorised `get_k_kpts` | `crates/pyscf-pbc-df/**` |
| **17-11** | multigrid **v1** (`multigrid.py`) | `crates/pyscf-pbc-dft/src/multigrid/`, `crates/pyscf-kernels/src/multigrid*.rs` |
| **17-12** | multigrid **v2** (`multigrid_pair.py` + `pp.py` + `utils.py`) | `crates/pyscf-pbc-dft/src/multigrid/{pair,pp,utils}.rs` |
| **17-13** | this document | — |

`PBC-MASTER-PLAN §8.9` sized the phase at **eight** plans. It was an
undercount, not a compression: `basis.py`, `ktensor.py` and the entire
supermole workstream were missing, multigrid v1 and v2 were lumped into one
row, and 17-09's dependence on two unshipped phases was not recorded. §4
restates each correction with its evidence.

---

## §3 — The gates

Every row states the number originally recorded, the number 17-01 MEASURED,
and the number achieved.

### Gate A — the IBZ integers (EXACT, no tolerance)

| | originally | 17-01 measured | achieved |
|---|---|---|---|
| the six `nkpts_ibz` on a diamond-structure cell at `[16,16,16]` | (not stated) | `145 / 145 / 245 / 408 / 816 / 2052` | **EXACT on `si` and `diamond`** |
| the symmorphic controls `lif` / `he_fcc` | (not stated) | `145 / 145 / 145 / 408 / 408 / 2052` | **EXACT** |

`crates/pyscf-pbc-symm/tests/kpts_ibz.rs`. This is the phase's strongest
oracle-free gate: six integers, no tolerance, and 17-01 measured that they
travel with the space-group TYPE and not with the lattice constant, so `si`
(`a = 5.4306 Å`) and `diamond` (`a = 3.5668 Å`) reproduce upstream's own Si
set bit-for-bit. `finger(kpts_ibz)` deliberately is NOT gated — it scales as
`1/a` and would be testing the fixture, not the algebra.

**What it licenses:** the star search, the symmorphic branch, the
time-reversal fold and the `wrap_around` interaction are all correct as a
*combination*. **What it does not:** it says nothing about the transforms
(Gate B) — a `KPoints` with the right stars and a wrong `stars_ops` passes it.

### Gate B — the transforms, against ONE converged SCF

17-CONTEXT expected "≥1e-12 unconditionally". **That expectation was wrong**,
and 17-01 measured why: the floor is set by `cell.precision` (integral
screening), not by SCF `conv_tol`.

| quantity | 17-CONTEXT expected | 17-01 measured @ default `precision` | @ `precision = 1e-13` |
|---|---|---|---|
| `transform_dm` | ≥1e-12 | **4.481e-10** | 7.772e-15 |
| `make_rdm1(transform_mo_coeff)` | ≥1e-12 | **4.481e-10** | 7.772e-15 |
| `transform_mo_occ` | ≥1e-12 | **0** (exact) | 0 |
| `transform_mo_energy` | ≥1e-12 | **5.331e-11** | 5.107e-15 |
| `transform_1e_operator` | ≥1e-12 | **2.931e-11** | 5.551e-16 |
| `symmetrize_density` | ≥1e-12 | **6.607e-13** | 1.388e-16 |
| `mo_coeff` ELEMENTWISE (demonstration) | — | **2.296** | **2.429** |

**Gate B as restated and as met:** ≤1e-9 at default `cell.precision`, ≤1e-13
at `precision = 1e-13`. `crates/pyscf-pbc-symm/tests/kpts_transform.rs` and
`tests/symmetry.rs` gate it; 17-06's `tests/ktensor_ksymm_scf.rs` adds the
`KsymmArray` half against a REAL k-symmetric SCF (`oo` 8.255e-14, `ov`
3.842e-13, `vv` 3.318e-12).

**The elementwise `mo_coeff` row is evidence, not a footnote.** It is O(1) at
both precisions, because orbitals are defined only up to a unitary rotation
inside each degenerate subspace and every symmetric cell has degeneracies.
Every comparison in this phase that could have been written elementwise on
`mo_coeff` is written on the density, the energy or a projected block instead.

### Gate C — energy, port vs port, mesh PINNED

| | originally | 17-01 measured (upstream vs itself) | achieved (this port) |
|---|---|---|---|
| **`KRHF` ksymm vs full BZ, FFTDF** | `1e-14` / `1e-9` | 6.9e-14 … 2.8e-13 | **8.793e-14** (`si [2,2,2]`) |
| **`KRHF` ksymm vs full BZ, GDF** | " | 5.5e-10 … 9.4e-12 | **2.486e-10** (`si [2,2,2]`) |
| `KRKS` ksymm vs full BZ, FFTDF | " | 6.8e-14 … 1.5e-13 | **3.109e-14 / 2.842e-14** (both `use_ao_symmetry` branches) |
| `KRKS` ksymm vs full BZ, **GDF** | " | 1.8e-11 … 1.6e-10 | **1.432e-06 — NOT MET**, recorded, see §10 |
| DFT+U `E_U` IBZ vs full BZ | — | — | **6.939e-18** |
| `numint` unfolded-IBZ density vs full-BZ density | — | — | **1.054e-13** (tol 1e-11) |

`crates/pyscf-pbc-dft/tests/krks_ksymm.rs` for the DFT rows;
`crates/pyscf-pbc-mp/tests/kmp2_ksymm.rs` for the two `KRHF` rows, which each
print `|d E_scf|` beside the correlation-energy residual they exist to
qualify. **17-07 listed Gate C among its carry-overs and those two rows close
the `KRHF` half of it**, per DF route, on the same `si [2,2,2]` fixture 17-01
measured upstream on.

A number that is easy to mistake for Gate C and is NOT: 17-07 measured the two
`eig` ROUTES (`use_ao_symmetry` on vs off) agreeing at **1.703e-11** on
`e_tot` from two converged SCFs and at **9.186e-11** on every eigenvalue from
ONE Fock. That is a Schur's-lemma identity check inside the k-symmetric SCF,
not a symmetry-vs-no-symmetry comparison, and it is reported here as such.

### Gate D — energy, port vs upstream, per DF route

**NOT RUN as a Rust gate.** 17-01 measured upstream against ITSELF per route
(the table above); comparing this port's absolute energies against upstream's
needs the oracle harness, and the phase's own two-route comparisons are the
stronger test in every case where both exist. Recorded as a carry-over in §8
rather than claimed. What IS measured port-vs-upstream in this phase:

| quantity | port vs upstream |
|---|---|
| `KsymAdaptedRCCSD` `E_scf` on upstream's own He `[2,2,2]` fixture | **1.282e-06** |
| full-BZ `KRCCSD` `e_corr` on that fixture | **4.297e-11** |
| full-BZ `KRCCSD` `emp2` on that fixture | **2.993e-10** |
| the s2 k-quartet class list, sizes and 512-entry `bz2ibz` | **EXACT** |

### Gate E — multigrid, at the quadrature floor

Gate E is explicitly NOT at Gates C/D's level, and 17-01 measured why: v1 is
algebraically exact against FFTDF/`numint` (1e-12…1e-14) while **v2 carries a
definitional ~2.4e-08 (diamond) / 1.5e-07 (si) floor against FFTDF that does
not shrink with mesh.** That is upstream's own two-implementations-of-one-idea
gap, the same shape as Phase 14's GDF-vs-RSDF 4.5e-6 finding.

| | 17-01 measured (upstream) | achieved (this port) |
|---|---|---|
| v2 `get_j` vs FFTDF, diamond | ~2.41e-08 | **1.24e-08** |
| v2 `get_j` vs FFTDF, si | ~1.47e-07 | **6.80e-08** |
| v1 vs v2, diamond / si | 2.41e-08 / 1.47e-07 | **1.46e-08 / 7.41e-08** |
| `nr_rks(lda,vwn)` Δnelec / Δexc | — | **≤1.5e-6 / ≤7.9e-7** |

### The post-SCF gates (17-09)

**17-09 Task 0 — the prerequisite check, and its outcome.** The plan was
written `autonomous: false` with a gate that could stop it: *"If `KMP2` is
absent, this plan does not start. If `KRCCSD` is absent, the CC half is
deferred with a written statement of what is missing."* Re-run on 2026-09-07:

* `pyscf-pbc-mp` exports `Kmp2` with `e_corr`, `t2` and `make_rdm1`
  (`kmp2.rs:35-135`), and Phase 15's `ao2mo_7d` index contract is asserted in
  `tests/oracle_phase15.rs`. **PASS — the plan starts.** The contract was READ,
  not re-derived: `kmp2_ksymm` reuses `kmp2_kernel::{df_oovv, ao2mo_oovv}`
  and `build_lov` unchanged, so there is exactly one `(ia|jb)` index
  convention in the crate.
* `pyscf-pbc-cc` exports `Krccsd` (`kccsd_rhf.rs:774`) with a green oracle
  suite. **PASS — the CC half is NOT deferred**, and no stub files were
  created for it.

17-01 could measure only the MP2 half — `crates/pyscf-pbc-cc/src` was a
13-line stub on 2026-09-01. Phase 15 closed 2026-09-05 and Phase 16 shipped
`KRCCSD`, so both halves of 17-09 landed here and both are gated.

**`KMP2`** — `crates/pyscf-pbc-mp/tests/kmp2_ksymm.rs`, si `[2,2,2]`:

| quantity | 17-01 measured (upstream vs itself) | achieved |
|---|---|---|
| `e_corr` ksymm vs full BZ, **FFTDF** | 1.067e-09 (GDF, route-mixed) | **1.138e-10** (`|d E_scf|` 8.793e-14) |
| `e_corr` ksymm vs full BZ, **GDF** | 1.067e-09 | **5.632e-11** (`|d E_scf|` 2.486e-10) |
| the s2-class kernel vs the dense `nkpts^3` one, ONE reference | — | **3.278e-12** (pure re-association) |
| the `Lov` route vs the `ao2mo` route, ONE reference | — | **1.283e-12** |
| **`rdm1`, k-symmetric vs dense, ONE reference** | 5.028e-09 (upstream, two SCFs) | **0e0 — BIT-IDENTICAL** |
| `Tr(gamma)` weighted by `weights_ibz` vs `nelec` | — | **7.999999999965 vs 8 — 3.5e-11**; Hermiticity 2e-12 |
| `make_t2_for_rdm1` block count | — | **231 of 512** (2.2x fewer `ao2mo`/`Lov` contractions than the dense `t2`) |
| `e_corr` / `t2` at 1 vs 8 threads | — | **BIT-IDENTICAL** |

The `Tr(gamma)` row needs its own sentence, because the obvious version of
that test is wrong. **`Tr(gamma_k)` is NOT `nelec` at each k-point and upstream
does not satisfy that either** — measured on upstream's own KMP2 anchor, it
misses by `2.8e-2` on the very first k-point (`tests/kmp2.rs`), because the MP2
correction moves charge BETWEEN k-points. What holds is the **`weights_ibz`-
weighted** average, and under symmetry those weights are the star
multiplicities (`[0.125, 0.375, 0.5]` on `si [2,2,2]`), not `1/nkpts_ibz`. The
per-IBZ traces here are `8.0737 / 7.9804 / 7.9963` and only their weighted
mean is the electron count — so a mistaken `1/nkpts_ibz` would miss by `4e-2`,
not by a rounding error.

**The `rdm1` `0e0` row is not an accident of the fixture:** `make_t2_for_rdm1` builds only the `ki <= kj` blocks
that touch the IBZ and `_gamma1_intermediates` reconstructs the rest through
`t2[kj,ki,kb].transpose(1,0,3,2)`, so a wrong index in that reconstruction
would move it. It is the one place in 17-09 where a plausible-looking density
matrix could have come out of a wrong transpose, and no trace or Hermiticity
check could have seen it — which is why the test compares element-wise against
the dense route on ONE reference rather than checking invariants.

**`KRCCSD`** — `crates/pyscf-pbc-cc/tests/kccsd_ksymm.rs`, upstream's own He
`[2,2,2]` fixture (120 IBZ k-quartets of 512 k-triples, 4 IBZ k-points of 8):

| quantity | 17-01 measured | upstream measures | achieved |
|---|---|---|---|
| `e_corr` ksymm vs full BZ, ONE mean field | unmeasured (Phase 16 unshipped) | 4.250e-11 (corrected guard) | **4.838e-12** |
| `emp2` ksymm vs full BZ | " | — | **identical to 15 digits** |
| `|d t1|max` unfolded vs full BZ | " | 1.749e-08 | **1.262e-11** |
| `|d t2|max` unfolded vs full BZ | " | 1.392e-10 | **5.714e-12** |
| `e_corr` / `emp2` / `t1` / `t2` at 1 vs 8 threads | — | — | **BIT-IDENTICAL** |
| the RCCSD amplitude symmetry `t2[ki,kj,ka] == t2[kj,ki,kb]^T` over the WHOLE BZ | — | — | **1.804e-16** |

**This port's k-symmetric CC is an order tighter against its own full-BZ
route than upstream's is against its own** (4.838e-12 vs 4.250e-11), and three
orders tighter on `t1`. §6 records the upstream defect that accounts for part
of that.

The `1.804e-16` row is the file's strongest oracle-free statement. `t2` is
STORED at only 120 of 512 triples, so at the other 392 both sides of that
comparison come out of `transform_4d` — and it is the identity a wrong
`(label, trans)` pair breaks while leaving `e_corr` (a full contraction)
invariant.

---

## §4 — The scope corrections, restated with the evidence

`PBC-MASTER-PLAN §8.9` sized the phase at eight plans. Five things were wrong
with the table, and each was found before any code was written.

1. **An entire workstream was missing.** `ft_ao._RangeSeparatedCell` /
   `ExtendedMole` appears in none of `§8.9`'s eight rows, and **seven live
   Rust sites promised it to Phase 17 by number** (`gdf_builder/mod.rs:96`,
   `rsdf_builder/mod.rs:190`, `gdf_builder/eta.rs:196`, `gdf/jk.rs:36`,
   `gdf/jk.rs:243`, `mdf/mdf_jk.rs:80`, `rsjk.rs:31`). It became 17-10 and it
   closed six of the seven; see §7.
2. **`symm/basis.py` (161 l) appears nowhere in `§8.9`**, and it is not
   optional: `ksymm_scf_common_init` (`khf_ksymm.py:142`) defaults
   `use_ao_symmetry = True`, so it is the DEFAULT branch. It became 17-04.
3. **`lib/ktensor.py` was moved into this phase by `15-CONTEXT §1.1` and
   `§8.9` never recorded it.** It became 17-06, and 17-09's CC half is its
   only real consumer — exactly the reason 15-CONTEXT gave for moving it.
4. **`§8.9`'s 17-07 was two independent ports.** `multigrid/__init__.py`
   exports `MultiGridNumInt` (v1, 2 C entry points) and `MultiGridNumInt2`
   (v2, 12 C entry points + `pp.py` + `utils.py`), and it is **v2, not v1**,
   that `pyscf/pbc/grad/rhf.py:44` and `grad/uhf.py:40` `assert isinstance`
   on. They became 17-11 and 17-12, ordered last so that dropping them would
   cost nothing already built.
5. **`§8.9`'s 17-06 was blocked on two phases that had shipped no code.**
   `kmp2_ksymm` needs `KMP2` (Phase 15) and `kccsd_rhf_ksymm` needs `KRCCSD`
   (Phase 16); both were 13-line stubs when the phase was planned. The plan
   was written `autonomous: false` with a Task 0 that could stop it, and it
   did not start until both prerequisites shipped.

A sixth correction is worth recording because it went the other way:
**`§8.9` called `pyscf_spglib.py` "an optional bridge"; it is weaker than
that.** `SpaceGroup.backend` defaults to `'pyscf'` (`space_group.py:264`),
the native `search_point_group_ops` is a self-contained brute force over
19 683 integer matrices, and `space_group.py:288-290` warns that spglib
cannot handle `cell.dimension < 3` at all — which rules it out for the
`graphene` reference system. **It is deliberately not ported**; see §7.

---

## §5 — The speed rulings, and where they inverted

`17-CONTEXT §8` ruled that symmetry only pays for itself if `get_jk` exploits
it too, and adopted D-PBC-26. **Point 1 of D-PBC-26 was measured WRONG during
17-07** and the erratum is already in `17-CONTEXT §8`:

> The Coulomb density built from an IBZ k-list is `Σ rho_k / N_ibz` and the
> true one is `Σ w_k <rho_k>_star`, and `rho_k` is not point-group invariant.
> Measured `max |d veff| = 9.486e-2 Ha` on `si [2,2,2]`
> (`khf_ksymm.rs::ibz_only_get_jk_is_not_an_identity`).

The attainable bound is `nkpts / nkpts_ibz`, reached **bit-identically**
(`max |d| = 0e0`) by restricting the OUTPUT set (`kpts_band = kpts_ibz`)
rather than the sampling set. `JkRoute::IbzOnly` is kept, non-default and
behind a doc comment that says not to enable it, purely so the measurement
that disproves it stays reproducible.

**This is the phase's recurring shape: FOUR speed assumptions were tested and
all four failed in the same direction.**

| assumption | measured |
|---|---|
| D-PBC-26 point 1: an IBZ-sampled `get_jk` is an identity | **9.486e-2 Ha wrong** |
| upstream's multigrid is faster than reference `numint` | **0.18x–0.49x — SLOWER** |
| 17-05's star search parallelises | **0.99x** |
| 17-08's `numint` under symmetry saves work | it does full-BZ work **PLUS** an unfold |

---

## §6 — Defects the phase's own tests caught

Phases 13 and 14 caught four and six respectively. This phase caught the
following; **four of them are in upstream PySCF 2.12.1, not in the port.**

### 1. D-17-07-01 — `little_cogroup_ops` indexes the wrong space (UPSTREAM)

`little_cogroup_ops` is filled from `np.where(k2opk[ki] == ki)[0]`
(`kpts.py:112`) — indices into `k2opk`'s `2*nop` columns when time reversal is
on — but its consumer indexes `kpts.ops[iop]` directly (`basis.py:113`). At Γ
and every TRIM the second half is reachable, so **upstream raises
`IndexError`**. This port refuses with a typed `KptsSymmInputMismatch`, which
is how it was found. It is NOT patched around: every k-symmetric fixture in
the phase sets `time_reversal_symmetry = false` and says why. It gates
`use_ao_symmetry = true` + time reversal, which is upstream's DEFAULT
combination.

### 2. D-17-09-01 — `update_amps` guards on a STALE loop variable (UPSTREAM)

`kccsd_rhf_ksymm.py:112` reads

```
if kk == ka and kl == kc:
    tau_term_1 += einsum('ka,lc->klac', t1[ka], t1[kc])
```

inside a loop that unpacks `kk, kl, ki, kd = kq`. **`kc` is not bound there** —
it is left over from the preceding loop's `ki, kk, kc, kd = kq`. Momentum
conservation gives `kd = kk + kl - ki` and this term has `ka = ki`, so when
`kk == ka` the partner index is `kl` itself: the intended guard is `kk == ka`
alone, with `t1[kl]`.

**Measured** (`measurements/gate_kccsd_stale_kc.py`, upstream monkey-patched
against itself on its own He `[2,2,2]` fixture):

| `e_corr` | value | vs full-BZ `KRCCSD` |
|---|---|---|
| upstream ksymm, stale `kc` | `-0.007379123020832` | 6.917e-11 |
| ksymm with `kk == ka` | `-0.007379123132509` | **4.250e-11** |
| full-BZ `KRCCSD`, same mean field | `-0.007379123090006` | — |

The corrected guard moves `e_corr` by `1.117e-10` and lands **closer** to the
full-BZ answer. This port ships the corrected guard and gates against its own
full-BZ `KRCCSD` rather than against upstream's k-symmetric number, because
gating on upstream's would mean reproducing its defect.

### 3. D-17-09-02 — `use_ao_symmetry = True` returns NON-CANONICAL orbitals on a k-mesh with lower symmetry than the lattice, and `E_scf` cannot see it (UPSTREAM)

`make_kpts_ibz` snapshots `k2opk` **before** wiping the columns of operations
that move a k-point out of the mesh (`kpts.py:60-64`), and
`little_cogroup_ops` is then filled from that UNWIPED table (`:109-113`). So
`little_cogroup_ops[i]` carries the full little co-group of the **lattice** at
`kpts_ibz[i]`, including operations the k-mesh does not respect.
`symm_adapted_basis` builds `symm_orb` from that group, and `eig` then solves
`F c = S c e` **one irrep block at a time**.

Schur's lemma makes that exact only if `F` has no matrix elements between the
blocks. Here it does: `v_J` comes from `rho(r) = Σ_k rho_k(r)` over a k-mesh
that is not point-group invariant, so neither `rho` nor `v_J` is invariant
under the operations `symm_orb` was built from.

**The consequence is subtler than "the SCF converges somewhere else", and the
first version of this entry got it wrong.** A per-block solve of a Fock matrix
that is not block-diagonal still spans the right OCCUPIED SUBSPACE, so the
density and the total energy are unaffected — `|d E_scf|` measures
**5.329e-15**, i.e. the two routes agree. What the per-block solve does not
give is CANONICAL orbitals: the vectors returned are eigenvectors of the
projected Fock, not of `F`. Every post-SCF method whose denominators assume `F`
is diagonal in the MO basis is then wrong.

| `si [1,1,2]`, GDF | `use_ao_symmetry = false` | `use_ao_symmetry = true` |
|---|---|---|
| operations outside the k-mesh subgroup | 36 of 48 | 36 of 48 |
| `\|d E_scf\|` vs full BZ | **0e0** | **5.329e-15 — AGREES** |
| `max \|d mo_energy\|` vs full BZ | **0e0** | **8.229e-03** |
| `KMP2` `\|d e_corr\|` vs full BZ | **0e0** | **3.629e-04** |

**The middle row is the signature.** Orbital energies 8.2 mHa apart while the
total energy agrees to 5e-15 is exactly what a non-canonical basis of the SAME
occupied subspace looks like — it is not convergence noise, and it is not a
different variational minimum.

`si [2,2,2]`, whose mesh IS closed under the cubic group (**0 of 48**
operations outside the subgroup), is unaffected: 1.138e-10 WITH
`use_ao_symmetry = true`.

**This is why 17-07's own gate could not have caught it.**
`ao_symmetry_eig_matches_the_plain_route` compares `e_tot` between the two eig
routes and lands at 1.703e-11 — and `e_tot` is exactly the quantity that is
blind here. It took a post-SCF method to see it.

The port ships a `KPoints::ops_outside_kmesh_subgroup()` /
`little_cogroup_ops_outside_kmesh_subgroup()` detector (gated with no SCF at
all, `crates/pyscf-pbc-symm/tests/kpts_k4_s2.rs`, 0.15 s) and a WARNING in
`KsymAdaptedKrhf::kernel`. **A warning and not a refusal, deliberately:**
upstream has neither, every fixture this phase gates is a symmetry-closed
mesh, and turning it into a hard error is a behaviour change that belongs with
17-07's own suite rather than with the plan that found it. **It is a
carry-over (§10).**

It is also why the `si [1,1,2]` KMP2 gate runs with
`use_ao_symmetry = false` — and with it off, that gate is **`0e0` on `E_scf`,
on every `mo_energy` and on `e_corr`**, which says the s2 fold itself is exact
there (as it must be: `[1,1,2]` keeps no operation that moves a k-point, so
the fold is the dummy-index interchange alone, and `(pq|rs) = (rs|pq)` makes
that exact).

### 4. `MORotationMatrix.build` indexes with `-1` (UPSTREAM, latent)

`kpts.py:1156` takes `k2 = k2opk[ki]`, which contains `-1` for every operation
that does not map the k-mesh onto itself, and hands it to
`get_rotation_mat_for_mos` as an index. NumPy reads `-1` as "the last
k-point", so upstream silently builds a shaped, wrong rotation matrix; it
happens never to be read, because `make_kpts_ibz` wipes those columns and
`stars_ops` never names one. Rust indexes out of bounds and panics, which is
how this was found (`si [1,1,2]`, where most operations are wiped). This port
stores an EMPTY block there, so an accidental read is a loud
`KsymmShapeMismatch` rather than a wrong number.

### 5. D-17-08-01 — 17-08 Task 1's premise was factually wrong (PLAN)

The plan said all seven `numint.py` `isinstance(kpts, KPoints)` sites
"evaluate the density at the IBZ points, then symmetrize through
`kpts.symmetrize_density`". Verified against 2.12.1: **five** (`:328, :431,
:859, :908, :956`) unfold to the FULL BZ and run the ordinary path, **two**
(`:647`, `:779`) take `kpts_ibz` directly, and **`symmetrize_density` has no
caller in `pyscf/pbc/` outside its own unit test**. Caught by hitting the wall
the wrong premise implies — the density is built per grid BLOCK and
`symmetrize_density` rotates indices across the whole mesh.

### 6. D-17-08-02 — 17-08 Task 4's premise was wrong (PLAN)

The plan said DFT+U's local projectors "must be rotated with the space group".
They must not: upstream builds them DIRECTLY at the IBZ points and unfolds
nothing (`krkspu_ksymm.py:77`, `:93`).

### 7. D-17-08-03 — a Gate C precondition nobody had stated (PORT)

An IBZ-vs-full-BZ energy comparison is only valid **if the full-BZ solution is
itself symmetric**. The IBZ run is constrained to symmetric occupations; an
unconstrained full-BZ run is not. Measured on the KUKS open-shell fixture:
full-BZ occupations star-symmetric `alpha = true, beta = FALSE`, `|dE| =
4.533e-02` **with the IBZ energy LOWER** — a different, better state, physical,
not a defect. The energy gate now ASSERTS the precondition and is `#[ignore]`d
pending a fixture whose full-BZ solution is symmetric in both channels.

### 8. Three defects in 17-12's own kernels, all hidden by an OOM

`collocate_pair_level` materialised one f64 per `(image × monomial × ci × cj)
× grid point` — **192 GiB (si) / 231 GiB (diamond)**. Once the reductions were
fused into the kernel lane (peak RSS 0.46 GiB), the suite ran and found three
more: `p.coef·q.coef` applied twice on top of `E` (∫rho = 0.53 of 8.73 e), no
periodic wrap of the fused Gaussian (and, once added, a `[0,1)³` box on a grid
that is origin-centred in `[-0.5,0.5)`), and a polynomial-blind image
pre-screen that dropped negative far `p–p` terms.

### 9. `_s2_index`'s "completeness" is not a property upstream has (PORT test)

An early draft of `kpts_k4_s2.rs` asserted that a k-triple and its dummy-index
partner always share an s2 class. **They do not, and they do not upstream
either**: the refine pass (`kpts.py:236-273`) only searches representatives
whose four k-indices form the right multiset, so a partner in a star whose
representative has a different multiset is never found. Measured **48 of 512**
split on `si`/`diamond` `[2,2,2]`. The assertion was wrong, not the port; it
was replaced by a SOUNDNESS assertion (every class member is reachable from
its representative) plus the incompleteness pinned as an exact integer.

### 10. A refusal test that outlived its reason (PORT, caught by the run)

`kpts_ktuples.rs::make_k4_ibz_s1_quartets_conserve_momentum` asserted
`make_k4_ibz(cell, "s2").is_err()` — correct when 17-05 wrote it, because
`"s2"`'s only consumer was 17-09. When 17-09 shipped `"s2"`, that assertion
started failing, which is the system working: **the refusal could not outlive
its reason silently.** It was replaced by a shape check (`"s2"` is strictly
coarser than `"s1"`, and its three star-op fields come back empty as upstream's
return shape requires), with the values gated against upstream in
`kpts_k4_s2.rs`. `"s4"` still refuses, and that assertion stays.

---

## §7 — The performance ledger (17-13 Task 2)

**No row is blank.** A blank here would read as "never measured", which is
exactly the failure mode 17-01 exists to prevent, applied to speed instead of
precision.

| claim | 17-01's floor | measured (this port) | shipped? |
|---|---|---|---|
| `get_jk` fast path vs reference (17-07 Task 7) | `nkpts/nkpts_ibz` = 8x on `si [4,4,4]`; upstream's own full-vs-IBZ-subset ratio 223x FFTDF / 40x GDF | **the IBZ-SAMPLED route is not an identity: `max \|d veff\| = 9.486e-2 Ha`.** The attainable route is `kpts_band = kpts_ibz`, which is **bit-identical** (`max \|d\| = 0e0`) to the reference and costs `nkpts_ibz/nkpts` of the pair count | `JkRoute::Band` shipped; `JkRoute::Reference` is the default; `JkRoute::IbzOnly` kept non-default with the disproof |
| `KRKS`/`KUKS` `get_veff` under the same route (17-08 Task 5) | same | **already satisfied** — the DFT k-symmetric adapters have taken the band route since 17-08 Task 2 (`kpts_band = kpts.kpts_ibz`, `krks_ksymm.py:41-42`) | shipped |
| multigrid **v1** vs reference `numint` (17-11 Task 4) | upstream's own ratio **0.49x (diamond) / 0.21x (si)** — i.e. upstream's v1 is SLOWER | not separately re-timed; v1's accuracy gates are green | reference-only claim; **no speed win is claimed for v1** |
| multigrid **v2** vs reference `numint`, and v2 vs v1 (17-12 Task 5) | upstream's own ratio **0.39x (diamond) / 0.18x (si)** | **v2 `get_j` 21.8 s / 16.5 s vs reference 0.51 s / 0.46 s and v1 0.34 s / 0.37 s → 0.023x / 0.028x**, ~10x worse than upstream's own v2 floor | shipped for `isinstance` (Phase 18 asserts on it), **explicitly not for speed** |
| `KMP2` s2-class kernel vs the dense `nkpts^3` one (17-09 Task 3) | — | **2.43x** on `si [2,2,2]` (36 s2 classes of 512 k-triples), backend warmed first | shipped, default |
| `KRCCSD` k-symmetric `ao2mo` transform count (17-09 Task 4) | — | **75 transforms** for 120 IBZ k-quartets of 512 k-triples | shipped, default |

**Two of these rows say "measured slower and shipped anyway", and both say
why.** D-PBC-26 point 6 requires exactly that: *"If the fast path turns out
not to beat the reference route on the CPU backend […] say so with the
measured numbers and ship the reference route as the only one. Do not ship a
'faster' path that measured slower."* Multigrid is shipped because Phase 18's
`grad/rhf.py:44` branches on `isinstance(ni, MultiGridNumInt2)`, not because
it is fast; the `get_jk` fast path is shipped in the one form that is
bit-identical to the reference.

The `2.43x` MP2 row deserves one sentence of honesty: the class-count
reduction is `512/36 = 14.2x`, and the realised 2.43x is smaller because each
`(ki, kj)` group still rebuilds its own `oovv` table and the `Lov` table is
built over all `nkpts²` pairs on both sides. That is upstream's structure, not
a port shortfall, but it is not `14.2x` and this document does not say it is.

---

## §8 — The deferral ledger (17-13 Task 3)

`grep -rn "NotYetImplemented" crates/ --include=*.rs`, every hit that named
phase 17:

| site | expected disposition | actual |
|---|---|---|
| `pyscf-pbc-gto/src/kpts_mesh.rs:112` | closed by 17-05 | **CLOSED** — the stub is gone; the constructor moved to `pyscf-pbc-symm` (`kpts_mesh.rs:114` records it) |
| `pyscf-pbc-df/src/gdf_builder/mod.rs:96` | closed by 17-10 | **CLOSED** |
| `pyscf-pbc-df/src/rsdf_builder/mod.rs:190` | closed by 17-10 | **CLOSED** |
| `pyscf-pbc-df/src/gdf_builder/eta.rs:196` | closed by 17-10 | **CLOSED** |
| `pyscf-pbc-df/src/gdf/jk.rs:36, :243` | closed by 17-10 Task 4 | **CLOSED** (`gdf/jk.rs:664` records it) |
| `pyscf-pbc-df/src/mdf/mdf_jk.rs:80` | closed by 17-10 Task 4 | **CLOSED** |
| `pyscf-pbc-scf/src/rsjk.rs` | stays refusing, re-home | **RE-HOMED** — now `{ phase: 14 }` (the cintx `range_omega` gap, `D-PBC-24`) and `{ phase: 19 }` (the MPI variants, a named non-goal) |
| `pyscf-pbc-gto/src/supercell.rs:100` | stays refusing — upstream refuses too | **RE-HOMED** to `{ phase: 12 }`; upstream's own `pbc.py:784-785` refuses the same input |

**There is no remaining `NotYetImplemented { phase: 17 }` anywhere in
`crates/*/src`.** The only occurrences of the string "phase: 17" left in the
tree are three doc comments in `pyscf-pbc-df/tests/band_kpoints.rs` and one in
`gdf/jk.rs:664`, each recording what WAS refused and which plan closed it.

`pyscf-pbc-symm`'s `UnsupportedK4Symmetry` was narrowed rather than removed:
`sym = "s2"` shipped in 17-09 and only `"s4"` still refuses, because
**upstream's own tree has no caller for `"s4"`** (`kpts.py:284-292`) and this
port ships no number no oracle can check.

### The deliberate non-ports, so they read as decisions

| not ported | why |
|---|---|
| `symm/pyscf_spglib.py` | `SpaceGroup.backend` defaults to `'pyscf'`; the native path is upstream's own default and this phase implements it anyway; spglib cannot serve `cell.dimension < 3` (`space_group.py:288-290`), which rules out `graphene`. Recorded as a v2.1 nicety. |
| `KROHF` / `KROKS` / `KGKS` k-symmetric adapters | **no upstream module exists.** Not invented. |
| `KsymAdaptedKMP2.make_rdm2` | upstream's own `raise NotImplementedError` (`kmp2_ksymm.py:253-254`), gated by a test so the refusal cannot outlive its reason |
| `ktensor_direct = True` (`kccsd_rhf_ksymm.py:389-395`) | changes NO number — it trades memory for recomputation — and upstream's own tests never set it. Refused by name rather than silently ignored. |
| `KsymmArray`'s `sym = "s4"` k-quartets | no caller upstream (above) |

---

## §9 — Determinism and hygiene (17-13 Task 4)

| check | result |
|---|---|
| `xtask check-forbidden-paths` | **PASS** — 389 `.rs` files, no out-of-scope upstream PySCF imports |
| `xtask check-catch-unwind` | **PASS** — 723 `.rs` files, every `extern "C"` site pairs with `catch_unwind` |
| `xtask check-dependency-wall` | **PASS** — cubecl containment intact (ALG-06) |
| `xtask check-orphan-modules` | **PASS** — 379 source files, all reachable |
| `xtask check-forbid-lazy-static` | **PASS** |
| `xtask check-no-fma` (release-oracle) | **PASS** — 7 asm files scanned, no FMA mnemonics (FOUND-05), run 2026-09-07. Scope: its `SCAN_TARGETS` — `pyscf-{algebra,core,ccsd,kernels}`, `pyscf-pbc-{gto,df,tools}`. `pyscf-pbc-{symm,scf,mp,cc}` are NOT in that list, by the same argument the file already makes for `pyscf-pbc-dft`: every reduction that reaches a gated energy goes through `oracle_sum`/`oracle_dot`/`oracle_zdotu`, which live in the already-scanned `pyscf-algebra`. Phases 15 and 16 shipped `pyscf-pbc-mp` and `pyscf-pbc-cc` on the same reasoning. Adding them directly is a carry-over, not a gap this phase introduced. |
| `cargo clippy --all-targets` on `pyscf-pbc-{symm,mp,cc}` | **clean** — no warning originates in any file this plan added or edited (the remaining workspace warnings are pre-existing and in other crates) |
| **no `mod tests` in any `src/*.rs` the phase added** (AGENTS.md §2) | **PASS** — `grep -rn "mod tests"` over `pyscf-pbc-{symm,scf,dft,mp,cc,df}/src` returns nothing |
| `rustfmt` on the files this phase touched | clean (formatted per-file; the tree predates the installed rustfmt, so a workspace `cargo fmt` would rewrite unrelated files) |

### Two §9.3 items that need naming rather than a tick

* **Complex eigenvector phases canonicalised on the real part.** Satisfied in
  `eig` by construction, not by a check in this phase: `eig_symm_adapted`
  solves each irrep block with `pyscf_algebra::zeigh_gen`, which normalises
  `Cᴴ S C = I`, fixes the global phase and then applies
  `pyscf_core::canonicalize_signs` to the real part (`zeigh.rs:242-244`).
  **`eig_trs` is not shipped** (17-07 carry-over), so its half of this item has
  nothing to check.
* **Cross-platform Linux x86_64 vs macOS aarch64 `KRHF` to 1 µHa — NOT RUN.**
  This session has one machine. Recorded as not run rather than assumed; it is
  a carry-over of the same kind Phases 13-16 carried.

**What was re-run for this document, and what was not.** Precision matters
more here than a tick, so:

* **Re-run and green on 2026-09-07**: the five static `xtask` checks (all
  PASS, re-run AFTER every file this plan added, so `check-orphan-modules`'s
  379-file reachability count includes them); `check-no-fma` (7 asm files,
  PASS); `pyscf-pbc-mp --test kmp2_ksymm` (**11 tests**, green — though across
  two invocations rather than one: nine passed in the full sweep,
  `si_112_ksymm_matches_full_bz` FAILED there and is what found D-17-09-02,
  and it plus the new
  `use_ao_symmetry_is_unsound_when_the_kmesh_breaks_the_lattice_symmetry`
  passed on the rerun with the corrected fixture); `pyscf-pbc-cc --test
  kccsd_ksymm` (**4 tests**); and the six `pyscf-pbc-symm` targets this plan's
  edits can reach — `kpts_ibz`, `kpts_ktuples`, `kpts_transform`,
  `kpts_k4_s2`, `ktensor`, `ktensor_ksymm_scf` — **55 tests, 0 failures**. The
  last of those is the one that drives `MORotationMatrix::build`, the only
  pre-existing behaviour this plan changed.
* **NOT carried to completion**: `pyscf-pbc-symm --test basis`. Its
  `si_3x3x3` / `diamond_3x3x3` cases run converged `KRHF`s at
  `cell.precision = 1e-10` on 27 k-points and exceeded this session's 40-minute
  budget (`diamond_2x2x2` and `si_2x2x2` passed before the cut). That target
  exercises `symm_adapted_basis`, which this plan did not touch.
* **NOT re-run**: the `pyscf-pbc-scf` and `pyscf-pbc-dft` k-symmetric suites.
  This plan changed one thing in `pyscf-pbc-scf` — a `tracing::warn!` in
  `KsymAdaptedKrhf::kernel` (D-17-09-02), which cannot move a number — and
  nothing in `pyscf-pbc-dft`. Their figures above are quoted from 17-07's and
  17-08's own runs, with that attribution. Saying so is cheaper than implying a
  run that did not happen.

### Thread-count invariance, measured inside ONE process

Every one of these varies the worker count with an explicit
`rayon::ThreadPool` rather than an env-var sweep across processes — strictly
stronger, because both runs share every cached input.

| quantity | 1 vs 8 threads |
|---|---|
| `symmetrize_density` (17-05) | BIT-IDENTICAL |
| the `KPoints` unfolds (17-05) | BIT-IDENTICAL |
| multigrid v2 grid accumulations (17-12) | BIT-IDENTICAL at 1/2/3/8 |
| `KMP2` k-symmetric `e_corr` / `e_corr_ss` / `e_corr_os` (17-09) | **BIT-IDENTICAL** |
| `KRCCSD` k-symmetric `e_corr` / `emp2` / `t1` / `t2` (17-09) | **BIT-IDENTICAL** |

Upstream's own run-to-run spread on the same fixture is **2e-15** (2 ulps at
`|E| ≈ 7.5`, `measurements/gate_c_d_repro.out`) — a gate tighter than that
would be testing upstream's BLAS, not this port. Every Gate C/D number sits
comfortably above it.

---

## §10 — Carry-overs, each naming its next phase

### From 17-01 (resource-scoped by the measurement's time budget)

1. `lif` / `graphene` Gate C/D at PRODUCTION mesh rather than the 25³ cap.
   The capped numbers (`1.461e-04` and `6.391e-01`) are **mesh-cap artefacts,
   not measurements**: `lif`'s ionic electrostatics and `graphene`'s 20 Å
   vacuum both need a far finer mesh, and `graphene`'s symmetric run does not
   even converge at 25³. `he_fcc` at the same cap gives `2.779e-10`, which is
   what says the algebra is fine and the mesh is not.
2. `lif` / `he_fcc` / `graphene`'s mesh-unpinning demonstration at their true
   default mesh.
3. `diamond`'s remaining four cells of the full 2×2×2×2 Gate C/D grid.

### From 17-07

4. `eig_trs` — real `mo_coeff` at TRIMs. Its TRIM test is the only proof the
   branch is ever taken.
5. `get_rho`, the chkfile round-trip (including the k-count refusal),
   `to_khf`.
6. **KUHF / KGHF k-symmetric adapters.** `KROHF` has no upstream `*_ksymm`
   module and is not invented.
7. Plan 11-09's metal-occupancy test extended to the k-symmetric path.

   **Gate C's `KRHF` half is NO LONGER a carry-over** — 17-09's fixture
   measures it per DF route on `si [2,2,2]` (§3): FFTDF **8.793e-14**, GDF
   **2.486e-10**. Gate D remains one.

### From 17-08

8. **GDF Gate C is RUN and FAILS at 1.432e-06** (tol 1e-8; `e_full
   -7.774590218592`, `e_ibz -7.774588786147`; 1381 s) — ~3 orders above GDF's
   own measured floor and ~8 orders worse than FFTDF on the identical
   comparison. Recorded, not absorbed. The first hypothesis (the GDF
   `kpts_band` route rebuilds `_cderi`) was **tested and disproved**:
   `gdf_band_route_matches_the_direct_route` gives `max |dvj| = max |dvk| =
   0e0` on the same `_cderi` at a strict-subset band set, so **17-10 Task 4 is
   exonerated**. Leading hypothesis, UNTESTED: GDF's `_cderi` is fit on a
   k-set with no symmetry adaptation, so the full-BZ GDF solution may be
   slightly symmetry-broken — the D-17-08-03 class, GDF-specific because
   FFTDF is analytic. The check is `check_mo_occ_symmetry` on the full-BZ GDF
   solution.
9. The KUKS Gate C fixture — needs a system whose FULL-BZ solution is
   symmetric in both spin channels (D-17-08-03). The test exists and asserts
   the precondition; it is `#[ignore]`d, not deleted.
10. RSH under k-symmetry stays blocked on the Phase-14 `omega` carry-over
    (`gdf/jk.rs:674`, D-PBC-24).

### From 17-09

11. **D-17-09-02's disposition.** The port ships a detector
    (`KPoints::little_cogroup_ops_outside_kmesh_subgroup`) and a WARNING in
    `KsymAdaptedKrhf`. Whether `use_ao_symmetry = true` on a symmetry-breaking
    k-mesh should be a hard REFUSAL is 17-07's call, because turning it into
    one is a behaviour change in that plan's surface and needs that plan's
    suite re-run. The alternative fix — intersecting `little_cogroup_ops` with
    the k-mesh subgroup, which is what the physics wants — is a divergence from
    upstream and should be decided, not slipped in.

### From 17-12

12. `pp.rs`'s IBZ path is not yet connected to 17-05's `KPoints` (which now
    exists); multigrid v1/v2 are gamma-only and not yet selectable as an SCF
    `numint`.

### From 17-09 (continued)

13. **`kump2_ksymm` / `kuccsd_*_ksymm` do not exist upstream** and are not
    invented. `KsymAdaptedKMP2` is restricted-only, as upstream's is.
14. `EriRoute::Ao2mo` reproduces upstream's literal integral choice for the
    k-symmetric MP2 kernel; `EriRoute::MatchFullBz` (the default) takes the
    route the full-BZ kernel would. Both are gated against each other; the
    only thing carried over is a systematic per-DF-route comparison against
    upstream's absolute numbers, which is Gate D's carry-over above.

### Still open from earlier phases, unchanged by this one

15. `ft_ao._RangeSeparatedCell` / `ExtendedMole` — **17-10 closed six of the
    seven Rust sites that promised it**; `rsjk` stays refusing because its
    SECOND blocker (a screened periodic 4-centre `int2e` driver,
    `PBCVHF_direct_drv1`) is untouched and "the screening IS the algorithm"
    (`rsjk.rs:41-52`). Re-homed to `{ phase: 14 }` / `{ phase: 19 }`.

---

## §11 — Worth reporting upstream

Four findings in PySCF 2.12.1, in descending order of consequence.

1. **`kccsd_rhf_ksymm.update_amps` guards on a stale loop variable**
   (`:112`), dropping a T1 contribution worth **1.117e-10** on upstream's own
   test fixture and leaving the k-symmetric answer FURTHER from the full-BZ
   one than the corrected version (6.917e-11 vs 4.250e-11). §6.2 has the
   derivation and the measurement; `measurements/gate_kccsd_stale_kc.py`
   reproduces it against upstream alone, with no Rust in the loop.
2. **`little_cogroup_ops` indexes `k2opk`'s doubled column space while
   `symm_adapted_basis` indexes `ops`** (`kpts.py:112` vs `basis.py:113`), so
   `use_ao_symmetry = True` with `time_reversal_symmetry = True` raises
   `IndexError` at Γ and every TRIM — upstream's own DEFAULT combination.
   §6.1.
3. **`use_ao_symmetry = True` returns NON-CANONICAL orbitals on a k-mesh with
   lower symmetry than the lattice** — `little_cogroup_ops` is filled from the
   UNWIPED `k2opk` (`kpts.py:60` then `:109-113`), so it names operations the
   mesh does not respect and the per-irrep `eig` solves a Fock matrix that is
   not block-diagonal in that decomposition. The occupied SUBSPACE — and
   therefore `E_scf` — is unaffected (measured `5.329e-15`), so upstream's own
   test of that quantity is blind to it; the orbitals are not Fock
   eigenvectors, and `KMP2`'s `e_corr` moves by **3.629e-04** on `si [1,1,2]`
   (36 of 48 operations outside the subgroup) against **0e0** with the
   constraint off. §6.3.
4. **`MORotationMatrix.build` passes `-1` as a k-point index**
   (`kpts.py:1156`), silently building a rotation onto the wrong k-point for
   every wiped operation. Latent — the result is never read — but it is a
   real `-1`-as-index. §6.4.

A fifth, smaller: **`kmp2_ksymm.make_rdm1` zips `nkpts_ibz` density blocks
against the FIRST `nkpts_ibz` entries of a full-BZ `padding_k_idx` list**
(`:135`). Those are the padding patterns of BZ points `0..nkpts_ibz`, not of
the IBZ representatives. The two agree whenever every k-point has the same
`nocc`/`nmo` — which is every fixture upstream tests — and disagree otherwise.
This port indexes `padding_idxs[ibz2bz[i]]` and says so in
`kmp2_ksymm`'s module doc.

---

## §12 — Reconciliation (17-13 Task 6)

`17-01` rewrote the gates at the START of the phase. This task reconciles the
three planning files with what was actually ACHIEVED, in one commit, because a
gate that appears in only two of the three is the exact failure 17-01 existed
to prevent.

| file | what changed |
|---|---|
| `ROADMAP.md` | the Phase-17 checkbox is ticked; the 2026-09-02 status rollup is replaced by the closure rollup, which carries 17-09's two halves, D-17-09-01, the third upstream defect, and the four-for-four speed finding |
| `PBC-MASTER-PLAN.md` | `§8.9`'s table gains a per-plan STATUS column; `§7`'s Phase-17 row records Gate D's disposition (not run as a Rust gate; 17-01's upstream-vs-upstream numbers stand, and the port's own two-route comparisons are the stronger test where both exist); D-PBC-26 already carries its erratum and now carries its disposition |
| `STATE.md` | current position, plan counts, and the honest status of the two gates NOT met (17-08's GDF Gate C at 1.432e-06, and Gate D) |

**Two gates in this phase are NOT met, and both are recorded as not met.**

1. **17-08's GDF Gate C, `1.432e-06` against a 1e-8 tolerance.** Recorded, not
   absorbed into a looser tolerance. The first hypothesis was tested and
   disproved; the leading one is untested and named.
2. **Gate D (port vs upstream, per DF route) was not run as a Rust gate.**
   17-01 measured upstream against itself per route, and this phase's own
   two-route comparisons are tighter than any port-vs-upstream comparison
   could be given the `1.282e-06` mean-field residual measured in §3. It is a
   carry-over, not a pass.

---

## §13 — What this phase licenses, and what it does not

**Licensed.** A `KRHF` / `KRKS` / `KUKS` / `KRKS+U` / `KUKS+U` run over an
irreducible k-point set, on FFTDF, at an accuracy indistinguishable from the
full-BZ run (3.1e-14 … 1.7e-11 depending on the quantity). A `KMP2` and a
`KRCCSD` on top of it, each an order tighter against its own full-BZ route
than upstream is against its own. Multigrid v1 as an alternative `numint` at
machine precision, and multigrid v2 as the object Phase 18's `grad/rhf.py:44`
asserts on.

**Not licensed.**

* **k-symmetric SCF on GDF at production accuracy.** 17-08's GDF Gate C sits
  at `1.432e-06`. FFTDF is the route this phase gates.
* **`use_ao_symmetry = true` together with `time_reversal_symmetry = true`.**
  D-17-07-01 blocks upstream's own default combination; every k-symmetric
  fixture in this phase runs with time reversal OFF and says why.
* **Multigrid as a speed feature.** Measured 0.023x–0.028x of the reference
  route (upstream's own v2 is 0.18x–0.39x). It ships because Phase 18 branches
  on its type.
* **k-symmetric UHF / GHF / ROHF / MP2-unrestricted / UCCSD.** `KUHF`/`KGHF`
  are carry-overs; `KROHF`, `kump2_ksymm` and `kuccsd_*_ksymm` do not exist
  upstream and are not invented.
* **Any absolute energy claim against upstream.** Gate D was not run; §3's
  table gives what IS measured port-vs-upstream.

---

## §14 — Phase 18 is unblocked

`pyscf/pbc/grad/rhf.py:28,44` and `grad/uhf.py:28,40`
`assert isinstance(ni, MultiGridNumInt2)`. That object exists
(`pyscf_pbc_dft::multigrid::MultiGridNumInt2`), its kernels are gated 8/8 and
its host-side Gate E is 10/10 at 17-01's upstream floors. The dependency is
recorded in `PBC-MASTER-PLAN §8.10`.

The one thing Phase 18 should read before relying on it: **v2 is 0.023x the
reference route's speed on this port**, ~10x worse than upstream's own v2, so
any Phase-18 plan that budgets wall clock against a multigrid gradient should
budget it against that measurement rather than against "multigrid is the fast
path".

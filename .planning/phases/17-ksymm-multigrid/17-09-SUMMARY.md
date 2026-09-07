# 17-09 — `kmp2_ksymm` + `kccsd_rhf_ksymm` — SUMMARY

**Shipped 2026-09-07.** Both halves. The plan was written `autonomous: false`
with a Task 0 that could stop it; that gate was re-run and passed on both
counts, so nothing was deferred and no stub file was created.

---

## Task 0 — the prerequisite gate, re-run

| requirement | state on 2026-08-31 (when the plan was written) | state on 2026-09-07 |
|---|---|---|
| `pyscf-pbc-mp` exports a working `KMP2` with `e_corr`, `t2`, `make_rdm1` | 13-line stub | **`Kmp2` (`kmp2.rs:35-135`), oracle-green** |
| Phase 15's `ao2mo_7d` index contract asserted in its tests | — | **`tests/oracle_phase15.rs`** |
| `pyscf-pbc-cc` exports `KRCCSD` | 13-line stub | **`Krccsd` (`kccsd_rhf.rs:774`), oracle-green** |

**The `(ia|jb)` index contract was READ, not re-derived.** `kmp2_ksymm` reuses
`kmp2_kernel::{df_oovv, ao2mo_oovv}` and `lov::build_lov` unchanged, so the
crate has exactly one such convention and 15-CONTEXT §1.2's settlement stands.

---

## What landed

### `KPoints::make_k4_ibz(sym = "s2")` — the piece 17-05 left refusing

`crates/pyscf-pbc-symm/src/kpts.rs`. 17-05 shipped `"s1"` and refused `"s2"` /
`"s4"` because their only named consumer was this plan. `"s2"` folds the
dummy-index interchange `(ki,kj,ka,kb) <-> (kj,ki,kb,ka)` on top of the s1
stars in TWO passes (`kpts.py:219-235`, then the "refine" pass `:236-273`),
and the second is the one that is easy to get subtly wrong — it does not
compare representatives, it walks a whole s1 star looking for the interchanged
tuple.

`"s4"` still refuses: **upstream's own tree has no caller for it**
(`kpts.py:284-292`), so this port ships no `"s4"` number no oracle can check.

**Gated EXACTLY against upstream** (`measurements/gate_k4_s2.py` /
`gate_k4_s2.out`, `crates/pyscf-pbc-symm/tests/kpts_k4_s2.rs`): the class
list, the integer class sizes and the full 512-entry `bz2ibz`, on `si` AND
`diamond` at `[2,2,2]` with time reversal both ways — 36 s2 classes of 50 s1
classes of 512 k-triples — plus `si [1,1,2]` (8 -> 6), which isolates the
dummy-index fold from the star fold because that mesh keeps no k-mixing
operation.

**A wrong invariant, caught by the port and recorded rather than patched
around.** An early draft asserted that a triple and its dummy-index partner
always share an s2 class. **They do not, and they do not upstream either**:
the refine pass only searches representatives whose four k-indices form the
right multiset. Measured **48 of 512** split on `si`/`diamond` `[2,2,2]`. The
assertion was replaced by a SOUNDNESS one (every class member is reachable
from its representative under the s1 star composed with at most one
interchange) plus the incompleteness pinned as an exact integer, so a refine
pass that diverges from upstream's in EITHER direction fails.

### `kmp2_ksymm.rs` — `KsymAdaptedKMP2`

`crates/pyscf-pbc-mp/src/kmp2_ksymm.rs` (+ `tests/kmp2_ksymm.rs`, 10 tests).

The design point worth stating first: **the k-symmetric KMP2 is not an IBZ
calculation the way the k-symmetric SCF is.** `KMP2.__init__` (`kmp2.py:715-722`)
UNFOLDS `mo_energy` / `mo_coeff` / `mo_occ` to the full BZ and sets
`nkpts = kpts.nkpts`; the only thing symmetry changes is WHICH `(ki,kj,ka)`
triples are evaluated. `unfold_kscf_result` is therefore a public, explicit
step rather than something hidden in a constructor, and handing the adapter an
IBZ-length result is an error, not a plausible wrong number
(`unfold_produces_a_full_bz_reference` gates that).

**Two deliberate deviations, both stated in the module doc:**

1. **The ERI route.** `kmp2_ksymm.py:46` takes `mp._scf.with_df.ao2mo`
   unconditionally while `kmp2.kernel:114-125` branches on `with_df_ints` and
   uses the `Lov` route for GDF. So upstream's own "ksymm vs full BZ" number on
   a `density_fit` reference is a comparison of two INTEGRAL ROUTES as much as
   of two k-sets — very likely most of the `1.067e-9` 17-01 measured on `si`,
   against `3.096e-16` on a He cell where the two routes agree far more
   closely. This port defaults to `EriRoute::MatchFullBz`, which route-matches
   both sides; `EriRoute::Ao2mo` reproduces upstream's literal choice and the
   two are gated against each other on ONE reference.
2. **`make_rdm1`'s padding index.** Upstream zips `nkpts_ibz` density blocks
   against the FIRST `nkpts_ibz` entries of a full-BZ `padding_k_idx` list
   (`:135`) — the padding patterns of BZ points `0..nkpts_ibz`, not of the IBZ
   representatives. They agree whenever `nocc`/`nmo` is uniform, which is every
   fixture upstream tests. This port indexes `padding_idxs[ibz2bz[i]]`.

**Measured** (`si [2,2,2]`, both DF routes, `release-oracle`):

| gate | result |
|---|---|
| `e_corr` ksymm vs full BZ, FFTDF | **1.138e-10** (`\|d E_scf\|` 8.793e-14) |
| `e_corr` ksymm vs full BZ, GDF | **5.632e-11** (`\|d E_scf\|` 2.486e-10) |
| s2-class kernel vs dense `nkpts^3` kernel, ONE reference | **3.278e-12** (pure re-association) |
| `EriRoute::MatchFullBz` (`Lov`) vs `EriRoute::Ao2mo`, ONE reference | **1.283e-12** |
| `rdm1` k-symmetric vs dense, ONE reference | **0e0 — BIT-IDENTICAL** |
| `weights_ibz`-weighted `Tr(gamma)` vs `nelec` | **7.999999999965 vs 8** |
| `make_t2_for_rdm1` blocks built | **231 of 512** |
| wall clock, s2-class vs dense kernel | **2.43x** (backend warmed first) |
| `si [1,1,2]`, `use_ao_symmetry = false`: `E_scf`, every `mo_energy`, `e_corr` vs full BZ | **0e0 — BIT-IDENTICAL** |
| `e_corr` / `e_corr_ss` / `e_corr_os` at 1 vs 8 threads | **BIT-IDENTICAL** |

Upstream measures `1.067e-9` for the first row on the same cell, so this port
is an order tighter — and no longer SCF-bound, because both sides take the same
integral route.

The `0e0` row is the one that could not have been faked: `make_t2_for_rdm1`
builds only the `ki <= kj` blocks touching the IBZ and `_gamma1_intermediates`
reconstructs the rest through `t2[kj,ki,kb].transpose(1,0,3,2)`. That transpose
is the one place a wrong index yields a PLAUSIBLE density matrix, invisible to
any trace or Hermiticity check.

`make_rdm2` refuses, as upstream does (`kmp2_ksymm.py:253-254`), with a test
asserting the refusal so it cannot outlive its reason.

### `kccsd_rhf_ksymm.rs` + `kintermediates_rhf_ksymm.rs` — `KsymAdaptedRCCSD`

`crates/pyscf-pbc-cc/src/{ksymm_common,kintermediates_rhf_ksymm,kccsd_rhf_ksymm}.rs`
(+ `tests/kccsd_ksymm.rs`, 4 tests).

`ksymm_common` is the seam between `pyscf-pbc-symm`'s `KsymmArray` world
(`Vec<Complex64>`, IBZ block stores) and this crate's `ZArr` world (planar
`CTensor`, dense full-BZ arrays). Upstream's default is
`ktensor_direct = False` (`:389-395`): build on the `kqrts_ibz` quartets, then
`todense()`, because the amplitude equations read `t2[kk,kj,kc]` at arbitrary
triples. **This port ships that default path and refuses `ktensor_direct =
True` by name** — it changes no number and upstream's own tests never set it.

Three things in the intermediates have no analogue in the non-symmetric module
and are transcribed with the upstream line beside them:

* the rank-2 intermediates are **SYMMETRISED** over each quartet's STABILISER,
  not merely computed;
* `cc_Fvv` and `update_amps`' second T1 term **SCATTER** to a different
  k-point (`ka_prim = kpts.k2opk[ka, op_group]`), so using the stabiliser
  there would drop every contribution whose image leaves `ka`;
* `cc_Woooo` / `cc_Wvvvv` skip half their quartets via `_s2_index` and fill
  them by transposition — an exact halving, not an approximation.

**Measured** (upstream's own He `[2,2,2]` fixture — 4 IBZ k-points of 8, 120
IBZ k-quartets of 512 k-triples, 75 `ao2mo` transforms):

| gate | this port | upstream, same comparison |
|---|---|---|
| `e_corr` ksymm vs full BZ, ONE mean field | **4.838e-12** | 4.250e-11 (corrected guard) |
| `emp2` ksymm vs full BZ | identical to 15 digits | — |
| `\|d t1\|max` unfolded vs full BZ | **1.262e-11** | 1.749e-08 |
| `\|d t2\|max` unfolded vs full BZ | **5.714e-12** | 1.392e-10 |
| `e_corr` / `emp2` / `t1` / `t2` at 1 vs 8 threads | **BIT-IDENTICAL** | — |
| `t2[ki,kj,ka] == t2[kj,ki,kb]^T` over the WHOLE BZ (oracle-free) | **1.804e-16** | — |

---

## D-17-09-01 — an upstream defect that changes a number

`kccsd_rhf_ksymm.py:112` guards a T1 `t1 t1` term with
`if kk == ka and kl == kc`, inside a loop that unpacks `kk, kl, ki, kd = kq`.
**`kc` is not bound there** — it is left over from the preceding loop
(`:89`, `ki, kk, kc, kd = kq`). Momentum conservation gives
`kd = kk + kl - ki`, and this term has `ka = ki`, so when `kk == ka` the
partner index is `kl` itself: the intended guard is `kk == ka` alone, with
`t1[kl]`. When the stale `kc` happens to equal `kl` upstream's term is right;
the rest of the time it is simply missing.

**Measured against upstream ALONE** (`measurements/gate_kccsd_stale_kc.py`,
upstream monkey-patched against itself, no Rust in the loop):

| `e_corr` | value | vs full-BZ `KRCCSD` |
|---|---|---|
| upstream ksymm, stale `kc` | `-0.007379123020832` | 6.917e-11 |
| ksymm with `kk == ka` | `-0.007379123132509` | **4.250e-11** |
| full-BZ `KRCCSD`, same mean field | `-0.007379123090006` | — |

The corrected guard moves `e_corr` by `1.117e-10` and lands **closer** to the
full-BZ answer, which is the only reference either version is trying to
reproduce. **This port ships the corrected guard**, gates against its own
full-BZ `KRCCSD` rather than against upstream's k-symmetric number (gating on
upstream's would mean reproducing its defect), and reports the residual
against both.

## D-17-09-02 — `use_ao_symmetry = true` returns NON-CANONICAL orbitals on a symmetry-breaking k-mesh

Found by `si_112_ksymm_matches_full_bz` failing at **3.629e-04** while the same
comparison on `si [2,2,2]` sat at `1.138e-10`.

`make_kpts_ibz` snapshots `k2opk` **before** wiping the columns of operations
that move a k-point out of the mesh (`kpts.py:60-64`), and
`little_cogroup_ops` is filled from that UNWIPED table (`:109-113`). So
`little_cogroup_ops[i]` carries the full little co-group of the **lattice** at
`kpts_ibz[i]`, including operations the k-mesh does not respect.
`symm_adapted_basis` builds `symm_orb` from that group and `eig` then solves
`F c = S c e` one irrep block at a time — which Schur's lemma justifies only if
`F` has no matrix elements between the blocks, and here it does (`v_J` comes
from a density summed over a mesh that is not point-group invariant).

**The first version of this entry got the consequence wrong, and the
measurement corrected it.** The claim was "the SCF converges to a different,
symmetry-constrained solution". It does not: a per-block solve of a
non-block-diagonal Fock still spans the right OCCUPIED SUBSPACE, so the density
and the total energy are unaffected — `|d E_scf|` measures **5.329e-15**. What
is lost is CANONICALITY: the vectors returned are eigenvectors of the projected
Fock, not of `F`, and every post-SCF method whose denominators assume `F` is
diagonal in the MO basis is then wrong.

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

`si [2,2,2]`, whose mesh IS closed under the cubic group (**0 of 48**), is
unaffected: 1.138e-10 with `use_ao_symmetry = true`.

**Why nothing before this could have caught it.** 17-07's
`ao_symmetry_eig_matches_the_plain_route` compares `e_tot` between the two eig
routes and lands at 1.703e-11 — and `e_tot` is exactly the quantity that is
blind here. It took a post-SCF method to see it.

**Shipped:** `KPoints::ops_outside_kmesh_subgroup()` and
`KPoints::little_cogroup_ops_outside_kmesh_subgroup()` — gated with **no SCF at
all** (`crates/pyscf-pbc-symm/tests/kpts_k4_s2.rs`, 0.15 s) — plus a WARNING in
`KsymAdaptedKrhf::kernel`. **A warning and not a refusal, deliberately**:
upstream has neither, every fixture this phase gates is a symmetry-closed mesh,
and turning it into a hard error is a behaviour change that belongs with
17-07's own suite rather than with the plan that found it. Carried over as
such.

## A fourth upstream defect, found by a panic

`MORotationMatrix.build` (`kpts.py:1156`) takes `k2 = k2opk[ki]`, which
contains `-1` for every operation that does not map the k-mesh onto itself,
and hands it to `get_rotation_mat_for_mos` as an index. NumPy reads `-1` as
"the last k-point", so upstream silently builds a shaped, wrong rotation
matrix; it happens never to be read, because `make_kpts_ibz` wipes those
columns and `stars_ops` never names one. Rust indexes out of bounds and
panics, which is how this was found (`si [1,1,2]`, where most operations are
wiped). This port stores an EMPTY block there, so an accidental read is a loud
`KsymmShapeMismatch` rather than a wrong number.

---

## Fixture traps this plan hit, so nobody re-derives them

* **`test_kccsd_ksymm.py`'s He cell is in ANGSTROM.** It writes
  `He.a = np.eye(3)*2.` and never sets `He.unit`, so PySCF's default applies.
  Reading it as Bohr gives a cell 1.89x smaller, a `[19,19,19]` mesh instead
  of `[35,35,35]`, and `E_scf = -1.402208` instead of `-2.093152`. The test
  asserts the mesh AND the mean-field energy for exactly this reason.
* **This port's `KRHF` is `1.282e-06` from upstream's on that fixture**, the
  same class of mean-field residual 16-01 found on diamond. Every comparison
  that could be contaminated by it is either driven from ONE mean field or
  gated loosely and reported.
* **Timing the first `kernel()` call on a GDF object times the `_cderi`
  build** (~90 s on `si [2,2,2]`), not the kernel. An earlier version of
  `cost_is_reported` reported a `0.00x` ratio for that reason; it now warms
  the backend first.

---

## Carried over

* `kump2_ksymm` and `kuccsd_*_ksymm` **do not exist upstream** and are not
  invented. `KsymAdaptedKMP2` is restricted-only, as upstream's is.
* A systematic per-DF-route comparison of the k-symmetric post-SCF energies
  against upstream's ABSOLUTE numbers — Gate D's carry-over, owned by 17-13's
  ledger rather than by this plan.

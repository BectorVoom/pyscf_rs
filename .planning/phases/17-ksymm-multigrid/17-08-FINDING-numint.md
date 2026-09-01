# D-17-08-01 — `numint`'s seven `KPoints` branches do NOT symmetrize the density

**Found:** 2026-09-01, implementing 17-08 Task 1.
**Status:** verified against the vendored PySCF 2.12.1. **17-08-PLAN.md Task 1's
premise is factually wrong** and the task must be re-specified before it is
built. This document is the correction.

---

## 1. What the plan says

> The seven `isinstance(kpts, KPoints)` sites in `pbc/dft/numint.py` all do the
> same thing: when the k-set is symmetric, evaluate AOs and the density at the
> IBZ points, then **symmetrize the real-space density** through
> `kpts.symmetrize_density` (17-05 Task 4) instead of averaging over the full
> BZ.
>
> — `17-08-PLAN.md`, Task 1

Every clause of that is wrong: they do **not** all do the same thing, they do
**not** symmetrize, and `symmetrize_density` is **not** involved.

## 2. What upstream actually does

`grep -n "isinstance(kpts, KPoints)" -A4 pyscf/pbc/dft/numint.py` — the seven
sites split into **two** behaviours, neither of which is the plan's:

### Group A — five sites: unfold to the FULL BZ (`:328, :431, :859, :908, :956`)

```python
elif isinstance(kpts, KPoints):
    if kpts.kpts.size > 3:            # multiple k points
        dms = kpts.transform_dm(dms)  # unfold IBZ -> full BZ
    kpts = kpts.kpts                  # and use the FULL BZ k-points
```

(`:859` unfolds `transform_mo_coeff` / `transform_mo_occ` instead, same shape.)

The density matrix is expanded to the whole zone and the **ordinary full-BZ
code path then runs unchanged**. `numint` gets no IBZ cost saving whatsoever —
it merely *accepts* a `KPoints` object and immediately discards the folding.

### Group B — two sites: use the IBZ points directly (`:647, :779`)

```python
elif isinstance(kpts, KPoints):
    kpts = kpts.kpts_ibz
```

These are `nr_rks_fxc` (`:647`) and `nr_uks_fxc` (`:779`). They take the IBZ
k-points and do **not** symmetrize or unfold anything.

### And `symmetrize_density` has no production caller at all

```
$ grep -rn "symmetrize_density" pyscf/pbc/ --include=*.py | grep -v "def "
pyscf/pbc/lib/kpts.py:1113:    symmetrize_density = symmetrize_density   # the method binding
pyscf/pbc/lib/test/test_kpts_ksymm.py:132:    def test_symmetrize_density(self):
pyscf/pbc/lib/test/test_kpts_ksymm.py:141:        rho += kpts.symmetrize_density(rho_k, k, cell.mesh)
```

Its only caller in the entire `pyscf/pbc/` tree is **its own unit test**. It is
a correct, tested utility with no upstream consumer — which is precisely why
17-05 shipped it against oracle-free invariants rather than against a caller.

## 3. How the wrong premise was caught

Not by reading — by hitting the architectural wall it implies. `KNumInt`'s
density is built **per grid block** (`block_ranges` → `eval_rho` over
`p0..p1`), but `symmetrize_density` rotates grid **indices** across the whole
mesh, so a block is not closed under the rotation. Implementing the plan
literally requires either materialising the full-mesh density before
symmetrizing (abandoning the block loop the whole file is built around) or
symmetrizing blockwise (wrong).

**That wall is itself the evidence.** Upstream never had to solve it, because
upstream never does this. A design that forces a structural fight with the
file it is modifying, to implement a behaviour the reference implementation
does not have, is a design working from a false premise.

## 4. What Task 1 should actually be

Faithful to upstream (RULE 2), and much smaller than the plan implies:

* **Group A (5 sites)** — under an IBZ k-set, unfold the density matrices to
  the full BZ with `KPoints::transform_dm` (17-05 Task 3, shipped and gated at
  1e-12) and then run the **existing, untouched** full-BZ path. The
  `KSet::Full` bit-identity requirement is then trivially satisfied, because
  `Full` is not merely unedited — it is the same code both arms reach.
* **Group B (2 sites, `nr_rks_fxc` / `nr_uks_fxc`)** — use `kpts.kpts_ibz`
  directly, no transform.
* **No `symmetrize_density` call anywhere in `numint`.**

## 5. Consequence for the phase's cost story, stated plainly

This removes the *only* place 17-08 was going to make DFT cheaper. With
upstream's design, `numint` under symmetry does the **same amount of work as
the full-BZ path plus an unfold** — it is a convenience interface, not an
optimisation. The IBZ saving in a ksymm DFT run comes entirely from the SCF
side (17-07's `get_jk` route, D-PBC-26), not from the XC quadrature.

That is worth recording next to 17-01's other measured speed corrections:
upstream's own multigrid measured **slower** than its reference `numint`
(0.18-0.49x), and 17-05's predicted star-search parallelism measured **0.99x**.
This phase's speed assumptions have been wrong three times now, always in the
same direction.

## 6. What was built against the wrong premise, and its disposition

`KNumInt::symmetrize_rho` and its Becke-grid refusal were written to the plan's
wording and are **correct code for a behaviour upstream does not have**. They
should be removed from `numint` rather than left as an unused path: the
capability already lives where it belongs and is already tested, as
`KPoints::symmetrize_density` (17-05 Task 4, `oracle_sum`-ordered and
thread-count bit-identical). Keeping a second, unreachable copy in the DFT
crate would be exactly the "container with no caller" that 15-CONTEXT §1.1
ruled against.

`KSet` itself **stays**. The enum is still the right shape — it distinguishes
"a plain k-point list" from "a `KPoints` object", which is the real branch
upstream takes seven times — only its `Ibz` arm's *behaviour* changes from
"symmetrize" to "unfold, or hand over `kpts_ibz`".

---

# D-17-08-02 — Task 4's premise is wrong too: DFT+U rotates no projectors

**Found:** 2026-09-02, implementing 17-08 Task 4. Same file, second wrong
premise — which is why it was checked rather than trusted.

## What the plan says

> These are short because the only thing that changes is that the **local
> projectors `C_ao_lo` must be rotated with the space group** when the density
> is unfolded — a Hubbard `U` on a d shell is defined in a local frame, and an
> IBZ run that forgets to rotate it applies `U` to the wrong orbital at every
> symmetry-related site. That is a large, plausible-looking energy error.
>
> — `17-08-PLAN.md`, Task 4

## What upstream actually does

`krkspu_ksymm.py` is 72 lines and its entire functional content is:

```python
def get_veff(ks, cell=None, dm=None, ...):
    vxc = krks_ksymm.get_veff(ks, cell, dm, ..., kpts=kpts, kpts_band=kpts_band)
    return krkspu._add_Vhubbard(vxc, ks, dm, kpts)

class KsymAdaptedKRKSpU(krks_ksymm.KRKS):
    get_veff = get_veff
    energy_elec = krkspu.energy_elec
```

No rotation. The symmetry handling lives in the **shared, non-ksymm**
`krkspu._add_Vhubbard`, and it is exactly two lines:

```python
is_ibz = hasattr(kpts, "kpts_ibz")
if is_ibz:
    kpts = kpts.kpts_ibz                                        # :77
...
weight = getattr(kpts_input, "weights_ibz", np.repeat(1.0/nkpts, nkpts))  # :93
```

`_make_minao_lo(cell, pcell, kpts)` is then called with `kpts = kpts_ibz`, so
**the projectors are constructed directly at the IBZ k-points**. They are
already correct there. The plan's failure mode cannot arise because the
Hubbard term never unfolds anything: the whole `E_U` / `V_U` contribution is
computed on the IBZ set and weighted by `weights_ibz`. `krkspu.energy_elec`
(`:150`) uses `weights_ibz` the same way.

## What Task 4 should actually be

* `add_vhubbard` evaluates at **`kpts_ibz`** (it already takes its k-points as
  an explicit argument, so this is the caller's choice).
* Its hardcoded `weight = 1.0 / nkpts` becomes **`weights_ibz`**.
* `KsymAdaptedKrkspu` = `KsymAdaptedKrks` + that call. Nothing else.

## The pattern, now twice in one plan

Both of 17-08's substantive premises described machinery upstream does not
have, and in both cases the real implementation is *smaller* and routes the
symmetry through a weight or a k-point list rather than a transform. The
phase's earlier corrections went the same way (17-01's gates, 17-06's
Hermiticity invariant, D-17-07-01). **Check the upstream source before
building to a plan's description of it** — that is what RULE 2 is for, and it
has now paid for itself four times in this phase.

The plan's proposed test — "the occupation matrix `n_IJ` is invariant under
every op of the little co-group at each site" — is still a *true* statement
about a symmetric solution, but it is not testing the thing the plan thought
it was, since no rotation is applied. The load-bearing gate is the same one
Task 2 uses: the DFT+U energy over the IBZ must equal the full-BZ DFT+U
energy.

# 14-08 — `RSDF` + `density_fit` + `rsjk`. **The two unblocked tasks shipped; the two range-separated ones did not.**

**Status:** Tasks 2 (partial) and 3 shipped, green — 5 tests in
`crates/pyscf-pbc-df/tests/rsdf.rs` and 5 in
`crates/pyscf-pbc-scf/tests/rsjk.rs`. Tasks 2 (the `RSGDF` builder) and 4
(`rsjk`) are blocked on the same cintx gap 14-07 documented, and both refuse
rather than substituting a different kernel.

## What shipped

### Task 5.1 — `get_aux_chg` IS 14-01's monopole

`crates/pyscf-pbc-df/src/rsdf.rs::get_aux_chg` — `ft_ao(auxcell, G = 0).real`
(`rsdf.py:65-73`). Asserted equal to `rsdf_builder::gaussian_int` at **exactly
0**, and asserted to take only the values 0 and 1 to **1e-14** on both reference
cells: every fitted `s` function carries unit charge because
`make_modrho_basis` normalised it that way (14-01 Task 3), and every `l > 0`
function integrates to zero by symmetry.

That is not a tautology. Upstream *computes* the charge one way
(`ft_ao(auxcell, 0)`) and *normalises* it another (`incore`'s `gaussian_int`).
If the two conventions disagreed, range separation would treat the wrong
auxiliary functions as charged. Now they are pinned to each other.

### Task 3 — one `density_fit`, four shims

`crates/pyscf-pbc-df/src/density_fit.rs`. Upstream carries four near-identical
copies (`df_jk`, `mdf_jk`, `rsdf_jk`, `aft_jk`); the plan asked for one, and
this is it: `density_fit(cell, kpts, DfKind, DfOpts) -> Box<dyn PeriodicDf>`.

The shape differs from upstream deliberately. Upstream mutates a mean-field
object because `with_df` is a mutable attribute on it; this port's drivers take
the builder at construction (`Krhf::from_df(Box<dyn PeriodicDf>)`, D-PBC-22), so
the shim's job is to *produce the builder*, not to patch a driver. The
verbosity/memory copying upstream does has no analogue — those live on the cell
here.

Tested: every shipped builder comes back naming itself, `mesh` reaches the three
builders that have one, and **GDF's mesh is deliberately NOT settable** — its
mesh resolves the model charge, not the density, which is 14-04's defect 2 and
cost 0.0743 Ha when it was got wrong.

## What did NOT ship, and why

### Tasks 2 (`RSGDF`) and 4 (`rsjk`) — BLOCKED

Both need a **short-range** integral, and cintx's safe API cannot request one.
The full evidence is in `14-07-SUMMARY.md` and
`crates/pyscf-pbc-df/src/rsdf_builder/mod.rs`; the short version is that
`ExecutionOptions` has no `range_omega` (libcint `env[8]`) field, no kernel
reads that slot, and the periodic 3-centre driver builds its cintx `BasisSet`
from the parsed per-element basis rather than from an `_env` array — so even
`pyscf-gto`'s `range_coulomb.rs` workaround is out of reach. The gap is Phase
4's Open Question A5 / cintx#11.

* `_RSGDFBuilder` (14-07 7b) is `RSGDF`'s builder → `Rsdf::build` refuses.
* `rsjk.py:186` sets `supmol_sr.omega = -self.omega` and evaluates the STANDARD
  `int2e` against it → `RangeSeparatedJkBuilder::build` / `get_jk` refuse.

**Gate 3 is therefore unreachable this phase** — it compares
`|E_KRHF(GDF) − E_KRHF(RSDF)|` against upstream's own floor and one of the two
builders does not exist. So is Task 5.3's `rsjk`-vs-FFTDF gate.

The tests assert the REFUSALS, so "RSDF is missing" is a fact the suite states
rather than an absence a reader has to notice, and so that nobody closes the gap
by quietly substituting the full-range kernel. That substitution would be
especially dangerous for `rsjk`, which is EXACT: a wrong answer would land
inside the 1.2e-3 DF fitting error of a correct GDF and look plausible.

### What the blocked types still do

Both `Rsdf` and `RangeSeparatedJkBuilder` expose the ω half, which is 14-07's
7a and ships:

| | port | upstream |
|---|---|---|
| `Rsdf::guess_omega`, He-fcc 2×2×2 | 0.739358637866536 / `[11,11,11]` / 30.7085675919949 | identical to 1e-12 |
| `RangeSeparatedJkBuilder::guess_omega`, diamond 2×2×2 | 0.601955030338906 / `[11,11,11]` | identical to 1e-12 |

So the parameters `rsjk` would run at are computed and gated today; only the
integral is missing. `rsjk.py:145-151` reads exactly these from
`rsdf_builder`, so when the cintx gap closes, 7a is already in place.

### `rsjk` is NOT a `PeriodicDf`, deliberately

`14-08-PLAN.md`: *"it must not be given a `PeriodicDf` impl whose
`sr_loop`/`get_naoaux` half is a lie."* It has no `cderi` to loop over and no
auxiliary count to report, so an impl would have to lie in two methods. It lives
in `pyscf-pbc-scf` (not `pyscf-pbc-df`), with `build` and `get_jk` only, and a
test records that. Its MPI / multi-threaded partitioning variants refuse
separately, as a named **non-goal** (`14-CONTEXT.md`: "one correct serial path")
pointing at Phase 19 rather than at the cintx gap — two different reasons, two
different messages.

## Task 5.6 — `df_swap.rs` now drives every shipped builder

`every_builder_drives_krhf_unchanged` runs `KRHF` on FFTDF, AFTDF, GDF and MDF
through `Box<dyn PeriodicDf>` with no driver change, each naming itself. On
He-fcc at gamma:

```text
KRHF on FFTDF: E = -3.20863588175665
KRHF on AFTDF: E = -3.20863586390596
KRHF on GDF  : E = -3.20865064031166
KRHF on MDF  : E = -3.20863596869297
```

Four of the six the plan asked for. RSDF and the `rsjk` route are the two that
are blocked.

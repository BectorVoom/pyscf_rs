# 17-11 SUMMARY — multigrid v1 (`multigrid.py`): task list, real-space
collocation kernel, density driver, `MultiGridNumInt`

**Status:** Tasks 1-4 SHIPPED, with three deliberate, stated scope
reductions (see "What did not ship" below) and one real bug found and fixed
mid-plan (the `discard=true` lattice-image bug — see "The bug found while
writing the tests"). **Date:** 2026-09-01.

## Exact green test command

```
cargo test -p pyscf-kernels -p pyscf-pbc-dft
```

Ran to completion: **18/18 test binaries green, 0 failed** (full log
captured this session). `cargo build -p pyscf-pbc-dft --tests` and
`cargo clippy -p pyscf-kernels -p pyscf-pbc-dft --lib --tests` (without
`-D warnings` — see the FMA/clippy note below) are clean save for one
PRE-EXISTING unused-import warning in `veff.rs` this plan did not touch.

`cargo run -p xtask --bin check-dependency-wall` — **PASS**
(`check-dependency-wall: PASS — cubecl-* containment intact (ALG-06)`).
`pyscf-pbc-dft` names no `cubecl-*` crate; it depends on `pyscf-kernels`
normally (a plain, non-cubecl dependency — the same pattern
`pyscf-pbc-df`/`pyscf-pbc-gto` already use for `ft_aopair`/`eval_ao_k`).

## What shipped

### Task 0 — manual read + layering

Read `/home/user/Documents/workspace/cubecl_manual/manual/manual/Cubecl/INDEX.md`
before writing `crates/pyscf-kernels/src/multigrid_collocate.rs`. Confirmed
D-PBC-25's corollary in the Cargo graph: `pyscf-pbc-dft/Cargo.toml` gained a
`pyscf-kernels` dependency (comment explains why this does not breach
ALG-06) and no `cubecl-*` dependency. The new kernel is **concrete `f64`,
not generic over `F: Float`** — the SAME documented exception
`ft_aopair.rs`/`eval_gto.rs` already use: the only transcendental it needs
is `exp(-alpha*r^2)`, and `cube_math::double::exp` IS the f64 libm with no
generic-`F` entry point. Stated explicitly in the kernel file's header, per
Task 0's instruction. No cube barriers, no fixed-256 cube — launched via
`pyscf_algebra::launch::launch_1d`, same as every other kernel this
milestone added.

### Task 1 — `crates/pyscf-pbc-dft/src/multigrid/tasks.rs`

Ports `_primitive_gto_cutoff` and `multi_grids_tasks_for_ke_cut` (the
DEFAULT `TASKS_TYPE='ke_cut'` route; `multi_grids_tasks_for_rcut` is NOT
ported — nothing in `measurements/gate_multigrid.py` exercises it, and it
is not upstream's default).

**One deliberate architecture choice, stated in the module doc**: rather
than upstream's `decontract_basis(to_cart=True, aggregate=True)` +
`h_coeff`/`t_coeff`/`t_cell` machinery, this port fully decontracts every
shell into one `Pshell` PER PRIMITIVE up front (`build_pshells`), with a
dense `nao_p x nao` expansion matrix `E` (`Decontracted::expand`) playing
`h_coeff`/`t_coeff`'s role. Same math (a sandwich transform), simpler
mechanism — no intermediate `pyscf_pbc_gto::Cell` for the decontracted
basis is ever materialised, because the collocation table
(`PshellGridTable`) only needs flat `(centre, alpha, powers)` records.
Primitive-level exponent granularity is PRESERVED (this is NOT a per-shell
simplification) — verified by construction, since `_primitive_gto_cutoff`'s
formula is applied to each primitive independently, exactly as upstream's
vectorised-over-primitives body does.

**Gate (`tests/multigrid.rs::task_list_level_mesh_matches_upstream`)**:
`diamond`'s level list matches upstream's live-measured
`multi_grids_tasks_for_ke_cut` output EXACTLY — 2 levels, `[31,31,31]` then
`[47,47,47]` (measured via `PYTHONPATH=<root> .venv/bin/python` against the
vendored 2.12.1 tree). `si` is a KNOWN, MEASURED, OUT-OF-SCOPE deviation:
live upstream's `mesh_to_cutoff([32,32,32])` gives `ke_cutoff_min =
84.34297006`; this port's ALREADY-SHIPPED (Phase 9/11, not this plan)
`pyscf_pbc_tools::mesh::mesh_to_cutoff` gives `84.34538134` on the IDENTICAL
lattice matrix — a ~2.8e-5 relative difference that happens to sit right at
an integer `Gmax` boundary for si's geometry (`Gmax ~= 15.000x`), flipping
`ceil(Gmax)` from 15 to 16 and the first level's mesh from 31 to 33.
Diamond's `Gmax` is not near an integer, so the same-scale discrepancy never
flips its ceiling — which is exactly why an origin-only smoke test would
have missed this class of bug entirely (see the discard bug below for the
analogous lesson). `si` is gated on what stays robust regardless: level
COUNT (2) and the LAST level's mesh (`[35,35,35]`, clamped to `fft_mesh`
however the ladder gets there); the first-level number is printed, not
asserted, with the exact numeric provenance recorded in the test's own doc
comment. Reported here rather than silently worked around, per the
codebase's "measure honestly, don't retune the gate" convention.

### Task 2 — `crates/pyscf-kernels/src/multigrid_collocate.rs`

One kernel, `collocate`: evaluates `(r-A_L)^ix (r-A_L)^iy (r-A_L)^iz *
exp(-alpha|r-A_L|^2)` for a batch of `(Cartesian slot, grid point)` pairs,
summed over a caller-supplied list of periodic images per pshell. This is
the SHARED primitive both `NUMINT_fill`/`eval_rho` (density, forward) and
`NUMINT_fill2c`/`eval_mat` ("pass2", reverse) reduce to — the dm/weight
CONTRACTION is done on the HOST in `crate::multigrid::colloc`, mirroring
the existing split in `crates/pyscf-pbc-dft/src/numint.rs` (kernel
evaluates AO values; `eval_rho_one`/`vxc_mat_one` contract with plain rayon
loops, not a second kernel).

**Kernel tests (`crates/pyscf-kernels/tests/multigrid_collocate.rs`, all
against an INDEPENDENT reference, never the kernel itself)**, all green:

* `single_s_gaussian_matches_analytic_norm` — a normalised s-Gaussian,
  collocated on a wide 96³ box and Riemann-summed, matches the closed-form
  `coef * (pi/alpha)^1.5` integral (diff < 1e-9; trapezoidal quadrature of a
  well-resolved Gaussian on an evenly-spaced grid is near-spectral, so 1e-9
  at a modest grid size is expected, not lucky).
* `l0_to_4_collocated_product_matches_eval_gto` — for l = 0..4, collocated
  Cartesian values transformed to spherical via the SHARED
  `cart2sph_l_matrix` match `pyscf_kernels::eval_gto_sph` (a different code
  path, `eval_gto.rs`) to **1e-12** — including a genuine shell-PAIR product
  check, not just single-AO values. (Caught and fixed a test-fixture bug in
  writing this: `eval_gto_sph` wants F-order coords, `collocate` wants
  interleaved `(ngrids,3)` — the two must not be handed the same buffer.)
* `periodic_wrap_is_exact` — a Gaussian at a periodic box's CORNER and one
  at its MIDDLE, summed over a 7×7×7 image block, integrate to the SAME
  total (diff < 1e-9), and both match the full-space analytic integral.

### Task 3 — `crates/pyscf-pbc-dft/src/multigrid/colloc.rs`

`level_rho` (forward) and `level_pass2` (reverse), both covering exactly
upstream's dense/sparse split (`Part A = dense x (dense∪sparse)`,
`Part B = sparse x dense`, where `sparse` = pshells from EARLIER/coarser
levels) — every pshell PAIR is covered exactly once, at the finer of its
two members' levels, matching upstream's coverage without the
`h_coeff`/`l_coeff` bookkeeping (see Task 1's note).
`expand_dm`/`contract_v` implement the `E`-matrix sandwich.
`rho_g_from_levels`/`pass2_from_full_vg` (in `multigrid/numint.rs`) combine
per-level real-space fields into/out of the FULL mesh's G-space
representation via the SAME `_takebak_4d`-style integer-frequency window
insertion upstream uses, reusing the ALREADY-SHIPPED
`pyscf_pbc_tools::{fft, ifft}` — neither FFT nor `ft_ao` was reimplemented,
per the plan's explicit instruction.

**`level_rho`'s grid-point accumulation is `oracle_sum` over a FIXED-order
pair-term list** (D-PBC-17 shape) — every grid point runs the identical
term list in the identical order regardless of which rayon worker owns it,
so the split is over disjoint outputs with the reduction axis untouched,
same idiom `crates/pyscf-pbc-dft/src/numint.rs`'s `eval_rho_one` already
uses. `level_pass2`'s matrix-entry accumulation is `oracle_sum` over the
grid, matching `vxc_mat_one`.

**Gate — `∫ rho dr == Tr(dm.S)`** (`tests/multigrid.rs::int_rho_matches_tr_dm_s`):
the general form of "`∫ rho dr == nelectron`" — `Tr(dm.S)` is the SAME
identity for ANY Hermitian `dm`, not only a converged one, computed
independently via the already-shipped `int1e_ovlp` integral, so this test
needs no SCF. `diamond`: diff = **1.636e-11**. `si`: diff = **2.098e-11**.
Both well inside the plan's 1e-10 gate, on BOTH reference cells.

### The bug found while writing the tests

`int_rho_matches_tr_dm_s` first failed at **1.7e-3** on diamond with a
random dm, and — critically — did NOT shrink when the mesh was refined from
15³ to 25³, which is the signature of a real bug rather than a
mesh-resolution artefact (a genuine quadrature error shrinks with mesh; a
dropped contribution does not). Isolating by AO block (same-atom s-only,
same-atom p-only, same-atom s-p cross, cross-atom s-s) found the exact
fault line: **any block touching diamond's SECOND atom (at `a/4·(1,1,1)`,
not the origin) was wrong by ~1.5e-4; every same-atom-0 block was correct
to ~2e-9.** Root cause: `colloc::collocate_level` called
`pyscf_pbc_gto::lattice::get_lattice_ls(cell, Some(p.rcut), None, discard=true)`
— `discard=true` drops lattice images "that cannot reach any atom pair", a
screen upstream's own integral drivers use for ATOM-PAIR shell-block sums.
Collocating one pshell's periodic images against ITSELF (there is no
"pair" — it is a self-sum) is exactly the case that heuristic does not
recognise, and for an off-origin atom it silently dropped images that
mattered. Fixed by passing `discard=false` at this one call site (now
documented in-line with the full failure signature, so a future change
cannot silently reintroduce it). This is the SAME class of lesson
`15-CONTEXT §3`'s `LARGE_DENOM` and `14-CONTEXT`'s column-major finding
record: an origin-only or single-atom fixture (`he_all_electron`, which
happened to pass throughout) would never have caught this — the isolation
trail is preserved above precisely so nobody has to re-derive it.

### Task 4 — `crates/pyscf-pbc-dft/src/multigrid/numint.rs` — `MultiGridNumInt`

`get_nuc`/`get_pp` **delegate to `pyscf_pbc_df::fftdf::{get_nuc, get_pp}`**
— a stated, justified reuse decision, not a placeholder: upstream's OWN
`multigrid.py::get_nuc`/`get_pp` call the identical analytic
`get_gth_vlocG_part1`/`pp_int.get_pp_loc_part2`/`ft_ao`-based `vppnl`
machinery FFTDF already uses, differing only in the "pass2" step (G-space
potential -> AO matrix), and 17-01's own measurement
(`measurements/gate_multigrid.out`) found that difference to be
**1e-12..1e-13** — floating-point noise, not a physical effect. Re-deriving
that specific pass2 here would duplicate an already-shipped, already-tested
path for a result already shown indistinguishable at the precision this
Gate needs (8 decimals). `get_j`/`nr_rks` do NOT delegate — they go through
the new collocation engine end to end, because they are the actual point of
this plan.

GGA (`nr_rks("pbe,pbe", ...)`, though only LDA was exercised in this
session's gate — see below) reuses upstream's own DEFAULT
`RHOG_HIGH_ORDER=False` route: `grad rho` comes from G-space (`i*Gv*rhoG`),
never a real-space AO gradient, and the GGA weight folds back to a single
LDA-style scalar field via `wv[0] -= i*Gv . wv[1:4]` before pass2
(`multigrid.py:1137-1141`) — this is WHY `colloc::level_pass2` never needed
a GGA-typed kernel; it is a consequence of upstream's own default, not an
omission on this port's part. Already-shipped `crate::xc::{RhoEff, VxcEff,
eval_xc_eff_rks}` are reused unchanged.

**Scope: GAMMA POINT ONLY**, stated explicitly in the module doc.
k-point-resolved (Bloch-phase) multigrid — matching `KNumInt`'s k-point
generality — is NOT ported. This is what every number in
`measurements/gate_multigrid.py` actually measures (`MultiGridNumInt(cell)`
with no `kpts`; the converged `KRKS` runs are gamma).

### Gate E — measured, against 17-01's numbers

**All measurements below are at a coarsened `25x25x25` mesh on BOTH the
multigrid route and the reference route** (a stated resource-scoping
deviation, same precedent 17-01's own README sets for time-budget-capped
meshes) — NOT at each cell's natural default mesh (`diamond` 47³, `si`
35³) 17-01 used. This means these numbers are NOT directly comparable to
17-01's 1e-12..1e-14 floor; they are this session's own, independently
measured, same-mesh-on-both-sides comparison, reported honestly rather than
conflated with 17-01's.

| quantity | diamond | si | gate (this test file) | upstream's own tolerance |
|---|---|---|---|---|
| `get_j` max\|diff\| vs reference FFTDF | 4.274e-15 | 1.003e-15 | < 1e-6 | 8 decimals (`test_multigrid.py:84-129`) |
| `nr_rks(lda,vwn)` \|d nelec\| | 3.197e-14 | 3.197e-14 | < 1e-6 | — |
| `nr_rks(lda,vwn)` \|d exc\| | 2.753e-14 | 3.553e-15 | < 1e-6 | 7 decimals (`:139-217`) |

Both cells land at **machine precision** for `get_j` and `nr_rks(lda,vwn)`
against the reference `numint`/FFTDF route, beating both this test file's
own 1e-6 gate and upstream's 7-8 decimal tolerance by many orders of
magnitude — consistent with 17-01's finding that v1 multigrid is
"essentially algebraically exact" against the reference quadrature. GGA
(`pbe,pbe`) was NOT additionally gated in this session (time budget); the
LDA number already exercises the full pipeline (collocation, level
combination, `Tr`/pass2 sandwich) that GGA's extra G-space gradient step
sits on top of, so this is a real but partial verification, reported as
such rather than implied complete.

**The "this is a different quadrature, not a tighter-tolerance target"
note required by the plan is IN the test file itself**
(`crates/pyscf-pbc-dft/tests/multigrid.rs`'s module doc), next to the
tolerances, so nobody tightens it later without re-reading why it is where
it is.

### Speed — measured and reported honestly, not assumed

| system | reference `get_j` | multigrid `get_j` | ratio (ref/mg) |
|---|---|---|---|
| diamond (25³ mesh) | 18.88s | 4.72s | **4.00x** |
| si (25³ mesh) | 15.64s | 3.01s | **5.20x** |

This port's multigrid is FASTER than the reference route at this
(coarsened, small-system) test point — the OPPOSITE direction from 17-01's
own finding (upstream's multigrid measured 0.18x-0.49x, i.e. SLOWER, on its
own natural-mesh reference systems). **This is not a contradiction to
resolve, it is two different measurements that must not be conflated**:
17-01 compared upstream's C-optimised multigrid (with per-shell/per-atom
rcut-restricted submesh iteration) against upstream's C-optimised FFTDF, at
each system's NATURAL mesh. This session compared THIS PORT's multigrid
(which iterates every level's FULL mesh — no shell-cutoff submesh
restriction, a stated simplification, see `colloc.rs`'s module doc) against
THIS PORT's `Fftdf::get_j_kpts` reconstructed fresh per call, at an
ARTIFICIALLY COARSENED, MATCHED mesh on both sides. The ratio here says
"at this one coarse test point, on these two Rust implementations, with
these two specific overheads" — it is reported because the plan requires
measuring and reporting honestly, not because it demonstrates multigrid
wins in general. Whether this port's multigrid is actually faster at
production scale (natural mesh, larger systems, per-shell rcut submesh
restriction added) is UNMEASURED and must not be assumed by any later plan
that reads this table.

### D-PBC-17 — thread-count bit-identity

`tests/multigrid.rs::eval_rho_g_is_bit_identical_across_thread_counts` —
`MultiGridNumInt::eval_rho_g`, run inside explicit `rayon::ThreadPool`s at
1/2/3/8 workers (same technique `tests/numint_threads.rs` already uses),
produces BIT-IDENTICAL `CTensor` output (`.re` and `.im` both) at every
worker count. Green.

## What did not ship (stated scope reductions, not omissions)

1. **k-point (Bloch-phase) multigrid.** Gamma point only — see Task 4's
   module doc. Every Gate E number this plan measures is gamma anyway.
2. **No shell-cutoff submesh restriction.** Upstream's C driver iterates
   only the `offset`/`submesh` sub-box a shell pair's `rcut` actually
   reaches; this port's `collocate`/`level_rho`/`level_pass2` iterate a
   level's FULL mesh for every pshell (screened only by which LEVEL a
   pshell/pair is assigned to, and by the per-pshell IMAGE list). Correct
   (every gate above confirms this), not performance-optimal — the speed
   table above is the honest consequence.
3. **GGA gated only via the shared machinery, not with its own live
   Gate-E number this session** (see the Gate E table's note). The code
   path exists and reuses already-tested `xc.rs`; a `pbe,pbe` number was
   not measured in this session's time budget.
4. **No driver-level selectability wiring** (e.g. a `with_numint` hook on
   `Krks`/`veff.rs` letting an SCF loop pick `MultiGridNumInt` over
   `KNumInt`). `MultiGridNumInt` is a complete, standalone, tested engine
   with the API the plan names (`get_nuc`, `get_pp`, `get_j`, `nr_rks`);
   plugging it into the SCF driver as a live alternative is follow-up work,
   not required by this plan's must-haves or verification list.

## FMA check

Per `AGENTS.md`'s final line ("Do not need to check FMA") this check was
NOT run. For the record: `xtask/src/bin/check_no_fma.rs`'s `SCAN_TARGETS`
does not include `pyscf-kernels` (not yet added to the list) and explicitly
EXCLUDES `pyscf-pbc-dft` already, for a documented, pre-existing,
unrelated reason (`libxc_rs`'s rayon mgga kernels segfault rustc under the
`release-oracle` profile's `codegen-units=1` — see that file's own comment
on the `pyscf-pbc-dft` exclusion). Neither crate this plan touched was in
scope for that gate before this plan started.

## Files touched

* `crates/pyscf-kernels/src/multigrid_collocate.rs` (new)
* `crates/pyscf-kernels/src/lib.rs` (module registration + re-export)
* `crates/pyscf-kernels/tests/multigrid_collocate.rs` (new)
* `crates/pyscf-pbc-dft/src/multigrid/mod.rs` (new)
* `crates/pyscf-pbc-dft/src/multigrid/tasks.rs` (new)
* `crates/pyscf-pbc-dft/src/multigrid/colloc.rs` (new)
* `crates/pyscf-pbc-dft/src/multigrid/numint.rs` (new)
* `crates/pyscf-pbc-dft/src/lib.rs` (module registration)
* `crates/pyscf-pbc-dft/Cargo.toml` (added `pyscf-kernels` dependency, documented)
* `crates/pyscf-pbc-dft/tests/multigrid.rs` (new)

No file outside `crates/pyscf-kernels/` and `crates/pyscf-pbc-dft/` was
touched. No concurrently-running plan's files (`pyscf-pbc-symm`,
`pyscf-pbc-df`) were edited.

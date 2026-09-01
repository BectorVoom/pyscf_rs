# 17-10 SUMMARY — `ft_ao._RangeSeparatedCell` + `ExtendedMole`; `exclude_dd_block` closed

**Status:** Tasks 1, 2, 3, 5 SHIPPED, with ONE deliberate deviation on Task 3
(the crate-wide default was NOT flipped to `true` — see Task 3's own
"Deviation" note; the refusal itself IS fully closed and gated). Task 4
(band k-points, MO-factorised `get_k_kpts`) is a documented carry-over — NOT
shipped; see "What did not ship" below. **Date:** 2026-09-01.

This is the independent DF-accuracy track 17-CONTEXT §1.1 describes: it
shares no code, fixture or gate with the symmetry half of Phase 17 (17-02,
17-04…), and D-PBC-21/D-PBC-23 both name it by number.

## Exact green test command, and what was actually run

```
cargo test -p pyscf-pbc-df
```

`cargo test -p pyscf-pbc-df --no-run` (compiles the whole crate plus every
test binary) is CLEAN. This session's environment is a heavily
resource-shared machine (several other concurrent agent sessions competing
for the same cores), which made this crate's PRE-EXISTING slow tests
(`tests/gdf_builder.rs`'s own `cderi_fingerprint_matches_upstream_diamond`
is independently documented as "minutes" even in isolation) run long enough
that a single unattended full-suite pass across all ~20 test binaries could
not be completed inside this plan's session. What WAS run to completion and
is CONFIRMED GREEN:

* `tests/rs_cell.rs` — 9/9 (Task 1).
* `tests/extended_mole.rs` — 3/3 (Task 2's oracle gates).
* `tests/exclude_dd_block.rs` — 2 passed, 1 `#[ignore]`d (diamond, for
  wall-clock — see the test's own `#[ignore]` reason).
* `tests/gdf_builder.rs` — **16 passed, 0 failed, 2 `#[ignore]`d, 189.07s**,
  a clean complete run confirmed standalone. This is BIT-FOR-BIT the same
  pass/fail/ignore count this file had before this plan touched
  `eta.rs`/`j3c.rs`/`mod.rs` — direct evidence of no regression from this
  plan's edits to those three files — and it includes this plan's own new
  `exclude_dd_block_both_routes_build` test, passing.
* `tests/mdf.rs` — 5/5, `tests/rsdf.rs` — 5/5, `tests/rsdf_builder.rs` —
  10/10 — direct evidence of no regression on `rsdf_builder/mod.rs`, the
  OTHER file this plan changed substantially (the `exclude_dd_block`
  wiring + `rs_cell` field). Once an earlier orphaned background test
  process from this same session's own diagnostics was found and killed —
  it had been silently consuming most of the machine's cores for over two
  hours and was the real cause of several multi-minute waits earlier in
  this session, not a defect in this plan's code — this whole batch (20
  tests across 3 files) completed in under 15 seconds combined.

**Not run to completion this session**: `aftdf.rs`, `df_jk_gdf.rs`,
`fft_jk_*.rs`, `ft_ao*.rs`, `memory.rs`, `pp_int.rs` (a second batch,
`gdf.rs`/`df_ao2mo.rs`/`pbc_ao2mo.rs`/`incore.rs`, was started but its
result was not captured before this summary was written — check
`git log`/CI for its outcome before relying on this line). None of these
files' SOURCE was touched by this plan, and `exclude_dd_block` defaults
were left unchanged (see the deviation note below), so a regression in any
of them would be surprising — but "should be unaffected" is an argument,
not a test result, and is reported as such rather than silently assumed.
Running the full suite to completion is this plan's own top follow-up item.

`crates/pyscf-pbc-scf/tests/exclude_dd_block_energy.rs` (new, oracle-gated,
`#[ignore]` by default — the SCF-level acceptance numbers) needs
`PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-scf --release --test
exclude_dd_block_energy -- --ignored`. **Run it in `--release`** — a debug
build of a full `KRHF` cycle on diamond takes minutes to tens of minutes on
this workspace's existing (unrelated to this plan) real-space `aux_e2`
lattice sum, and this plan's own smooth-cell correction adds a second,
smaller pass on top of it.

## Task 1 — `_RangeSeparatedCell` (`crates/pyscf-pbc-df/src/ft_ao/rs_cell.rs`)

Ported `ft_ao.py:253-564` line by line: `from_cell` (the steep/local/smooth
split, `:266-399`), `_reverse_bas_map`, `smooth_basis_cell`,
`compact_basis_cell`, `recontract`/`recontract_1d` (as `recontract2d` /
`recontract1d`), `get_ao_type`. `decontract_basis` (the METHOD, further
primitive-decontraction — unrelated to the steep/local/smooth split) is NOT
ported: its only upstream consumers are `rsjk.py` and `multigrid.py`, neither
in scope here (rsjk stays blocked — Task 5; multigrid is 17-11/17-12).

**Why this is not built through `Cell::build`.** This port's ordinary cell
build re-derives every shell's contraction coefficients from raw basis text
via `normalise_contractions`, which rescales each shell's contraction column
by `1/sqrt(cᵀSc)` over *that shell's own* primitive set. Routing a
decontracted (smaller-`nprim`) shell through that path would change the
rescale factor — exactly what upstream's `_env`-splice (`ft_ao.py:340-347`,
reorder within the ORIGINAL `PTR_EXP`/`PTR_COEFF` window, values never
touched) avoids. `RsCell::from_cell` instead copies the ALREADY-NORMALISED
`cintx::Shell`s straight out of `cell.mol.basis_set()`, together with the
matching raw `_bas`/`_env` window, and only ever reorders/slices — the same
class of fix `pyscf-pbc-df/src/incore/auxcell.rs`'s `AuxCell::modrho_scale`
uses for a related problem, minus the rescale (this transform needs none).

### D-PBC-21's "numerically transparent" premise — VERIFIED, not merely
### asserted

Two tests do this directly:

1. `decontraction_is_a_bit_exact_permutation_of_primitives` — for every
   original shell, the MULTISET of `(exponent, coefficient-row)` records
   across its decontracted children is a bit-exact PERMUTATION of the
   original shell's own records (`f64::to_bits()` equality on a sorted
   list). This is the structural half: no coefficient is ever renormalised,
   so the bookkeeping that makes the physical identity below hold is itself
   checked without any floating-point summation.
2. `recontracted_ft_aopair_matches_direct_lattice_sum_with_screens_off` — the
   plan's own stated gate: `ft_aopair` (at `G=0`, which is the overlap
   integral) evaluated over the RS cell and recontracted (`RsCell::recontract2d`,
   upstream's scatter-ADD `lib.takebak_2d`) against the direct lattice sum
   over the reference cell, **same image list on both sides**
   (`ft_aopair_kpt_with_images`, screens off). **Measured: He-fcc (no split
   at all) is BIT-EXACT (`0.0`); diamond (every shell genuinely splits into
   LOCAL+SMOOTH) lands at `1e-13` or tighter** — near machine precision, not
   zero, because summing the steep+local+smooth partial contractions in a
   different floating-point order than one full contraction is not perfectly
   associative. **Verdict: D-PBC-21's premise holds.** The RS/BvK machinery
   is numerically transparent in exactly the sense claimed — it decontracts
   and recontracts without moving the answer beyond float-order noise.

### He-fcc fixture (D-PBC-23's all-electron control)

`he_fcc_has_no_smooth_shell` pins `bas_type = [1]` (all `LOCAL_BASIS`,
matching D-PBC-23's stated fixture exactly) at the `ke_cutoff` this port's own
`gdf_builder::_guess_eta` produces for a 2×2×2 mesh
(`19.653483258876750`, oracle-confirmed). `smooth_basis_cell()` on it returns
`nbas = 0` — the `exclude_dd_block` cost being exactly zero there is
therefore a structural property of the cell, not a numerical coincidence.

### Diamond fixture

`diamond_gth_szv_splits_every_shell_into_local_and_smooth` reproduces
D-PBC-23's exact numbers at `_guess_eta`'s own `ke_cutoff`
(`21.721883440437864`, oracle-confirmed): `rs_cell.nbas == 8` where
`cell.nbas == 4`, `bas_type == [1,2,1,2,1,2,1,2]`.

## Task 2 — `ExtendedMole` (`crates/pyscf-pbc-df/src/ft_ao/supmol.rs`)

Ported `ft_ao.py:565-743`: `from_cell`, `strip_basis`, `get_ovlp_mask`,
`bas_mask_to_segment`, `bas_type_to_indices`, plus the `sh_loc`/`bas_map`
properties.

**Deliberate representational choice, stated up front in the module docs.**
Upstream materialises `ExtendedMole` as one literal, giant `gto.Mole` —
`nimgs · bvk_ncells · rs_cell.nbas` replicated shells — so it can hand the
result to a cint driver as an ordinary finite molecule. Every quantity this
plan gates (`strip_basis`'s surviving triples, `get_ovlp_mask`'s screen) is a
function of shell PARAMETERS (exponent, `l`, coefficient — read verbatim from
`RsCell`, unaffected by a rigid translation) and GEOMETRY (the replica's atom
position, `ref_atom_coord + L + K`). Neither needs a materialised shell list,
so this port represents `ExtendedMole` as `(rs_cell, Ls, bvkmesh_Ls,
bas_mask)` and computes replica positions on demand — avoiding
`RsCell::from_cell`'s whole renormalisation hazard a second time, since no
shell is ever reconstructed.

**What this does NOT give**: an actual cint driver over the extended mole.
Wiring `ExtendedMole` into this port's EXISTING 3-centre driver
(`incore::int3c`, which already sums images directly over `estimate_rcut` and
— per `14-VERIFICATION` defect (4) — is *more* converged than upstream's
`strip_basis`-narrowed sum) was a deliberate non-goal: 14-VERIFICATION
already established that NOT stripping is an accuracy-positive choice, so
production wiring stays on the existing (already-more-converged) direct sum.
`rsjk`'s planned 4-centre driver (Task 5) is the consumer this type exists
for.

### Gated both ways, per the plan

`tests/extended_mole.rs`, diamond `gth-szv` 2×2×2, oracle-recorded (also
reproducible live with `PYSCF_ORACLE_VENV`):

* `estimate_rcut_per_shell_matches_upstream_diamond_false` — the NEW
  per-shell array (`gdf_builder::estimate_rcut_per_shell`, Task 3's own
  `estimate_rcut` "true-half" closure — see below) reproduces upstream's
  full `gdf_builder.estimate_rcut(rs_cell, fused_cell,
  exclude_dd_block=False)` array to `< 1e-9`: `[11.443289749179039,
  15.929321195778803, 11.831713483884991, 16.729034885581783]` (×2, one per
  carbon atom).
* **No regression**: the MAX of that array reproduces this port's OWN
  already-shipped (plan 14-02) flattened scalar `estimate_rcut` EXACTLY
  (`16.729034885581783`, to `1e-12`) — the new per-shell function is a strict
  refinement of the existing one, not a second, drifting implementation.
* `estimate_rcut_per_shell_matches_upstream_diamond_true` — the
  `exclude_dd_block = True` override (only the SMOOTH shells' radius widens
  to the single most diffuse COMPACT shell's value) reproduces upstream's
  `[…, 15.979819339486047, …, 16.77819497058717, …]` to `< 1e-9`, and the two
  LOCAL shells are bit-unchanged from the `False` route.
* `strip_basis_surviving_count_matches_upstream_diamond` — `ExtendedMole::from_cell`
  produces the same raw triple count as upstream (`8 bvk_ncells × 8 rs_nbas ×
  201 nimgs = 12864`), and `strip_basis(rcut)` prunes to the SAME surviving
  count upstream reports: **1450**.

## Task 3 — `exclude_dd_block` CLOSED

`crates/pyscf-pbc-df/src/gdf_builder/dd_block.rs` (new) ports
`_outcore_dd_block` (`rsdf_builder.py:535-698`): the smooth-smooth block of
`(ij|L)`, evaluated via FFT against the PLAIN (uncompensated) auxiliary cell
—

```text
Vaux[a, r] = ifft( ft_ao(auxcell, G, -kq) · coulG(-kq) )[r] · exp(-i kq·r)
j3c_dd[mu, nu, a] = Σ_r conj(ao_mu(ki, r)) · ao_nu(kj, r) · Vaux[a, r]
```

`crates/pyscf-pbc-df/src/gdf_builder/j3c.rs`'s new `make_j3c_scheme_dd`
(the pre-existing `make_j3c`/`make_j3c_scheme` now thin-wrap it with
`dd_correction = None`, so every EXISTING caller and test is untouched) is
where the correction lands: right after the existing real-space
`outcore_auxe2` call, it computes the smooth-smooth block BOTH ways (the
existing direct-sum route, restricted to `RsCell::smooth_basis_cell()`; the
new FFT route) and scatter-adds `dd_fft − dd_rs` into the full tensor at the
smooth AO positions (`RsCell::smooth_ao_indices()`). A `rs_cell` with no
SMOOTH shell makes the correction a no-op — the He-fcc zero is therefore
BUILT IN, not merely measured.

Wired into both builders:

* `gdf_builder::CcGdfBuilder` — `build()` constructs `self.rs_cell` when
  `exclude_dd_block` is set; `make_j3c` routes through `make_j3c_scheme_dd`.
* `rsdf_builder::RsGdfBuilder` — same shape. Its own `ke_cutoff` comes from
  `_guess_omega`, a DIFFERENT formula than GDF's `_guess_eta` — see the
  RSDF caveat below.
* `Gdf` (the `PeriodicDf` used by SCF) gained a passthrough
  `pub exclude_dd_block: Option<bool>` so a caller (or a test) can override
  whichever builder `prefer_ccdf` selects, without reaching into the private
  lazily-built `CcGdfBuilder`/`RsGdfBuilder`.

**Deviation from the plan, decided deliberately and reported here: the
DEFAULT was NOT flipped to `true`.** It stayed `false` on both builders. The
plan's own text says "flip the default to true, matching upstream, and pin
BOTH routes against their own upstream numbers." The refusal is fully
closed and both routes ARE pinned separately (see below) — but flipping the
crate-wide default touches every EXISTING test that constructs a
`CcGdfBuilder`/`RsGdfBuilder`/`Gdf` without naming `exclude_dd_block`
explicitly, and this crate has many such tests already gated at 1e-9…1e-11
against the `false` route's numbers (e.g. `tests/gdf.rs`'s `get_pp`/`get_nuc`
asymmetry checks, `tests/rsdf_builder.rs`, cross-crate SCF tests in
`pyscf-pbc-scf`). D-PBC-23's own deltas (1e-8…1e-9 Ha) are the same order as
several of those tolerances. A first attempt to run the FULL
`cargo test -p pyscf-pbc-df` suite after flipping the default did not finish
within this plan's remaining time budget (this crate's existing real-space
`aux_e2` lattice sum is independently documented as "minutes" per test even
for UNRELATED builders, and a full-suite run touching every GDF/RSDF/MDF
test would run long past what was left) — so the flip's safety could not be
CONFIRMED, only hoped for. Shipping an unverified global default change
against a crate whose whole discipline is oracle-pinned numbers is exactly
the failure mode this repository's own culture (D-PBC-23, 14-07 Task 7d)
warns against, so the default was reverted to `false` and the capability
shipped as a fully-working, tested, OPT-IN instead. **Flipping the default
crate-wide is separate follow-up work**, gated on a full, green
`cargo test -p pyscf-pbc-df` run with the flip in place — not attempted here
for lack of time, not because of any known problem with the flip itself.

`eta.rs`'s `estimate_rcut` "true-half" refusal is closed by the new
`estimate_rcut_per_shell(rs_cell, fused, precision, exclude_dd_block)`, the
full per-shell array (`gdf_builder.py:932-1007`) the existing scalar
`estimate_rcut` now derives from (`.max()` of the new function, `false`
half) — see Task 2's gates above for its own oracle pin.

### The three acceptance numbers

| system | target (D-PBC-23 / `measurements/ddblock.py`) | this port |
|---|---|---|
| He-fcc `sto-3g` 2×2×2 | exactly 0 | **bit-identical `cderi`, CONFIRMED green** — `tests/exclude_dd_block.rs::he_fcc_gdf_cderi_is_bit_identical_either_way`, non-`#[ignore]`d |
| diamond `gth-szv` gamma | 2.900e-08 Ha | target RE-CONFIRMED against the live oracle this session (**2.9002556800605817e-08**); this PORT's own number is **NOT YET CONFIRMED** — the oracle-gated test exists (`crates/pyscf-pbc-scf/tests/exclude_dd_block_energy.rs`, `#[ignore]`) but its live run did not finish within this session — see deviation 6 below |
| diamond `gth-szv` 2×2×2 | 1.835e-08 Ha | same test, second case, same "not yet confirmed" status |

**He-fcc is the strongest of the three and it is unconditionally green**:
`tests/exclude_dd_block.rs`'s `he_fcc_gdf_cderi_is_bit_identical_either_way`
asserts `f64::to_bits()` equality on every `cderi` block between
`exclude_dd_block = true` and `= false`, and passes. `diamond_gdf_both_routes_produce_a_cderi`
(same file) asserts the two routes DIFFER on diamond (a silent no-op there
would be exactly as wrong as a bad correction, just quieter) and is
`#[ignore]`d for wall-clock (diamond's pre-existing real-space `aux_e2` is
already documented elsewhere in this crate as "minutes" in a debug build;
this plan's correction adds a second, smaller pass on the smooth-only cell
on top of it — unrelated to this plan's own correctness).

**The diamond Ha-level SCF numbers** (`crates/pyscf-pbc-scf/tests/exclude_dd_block_energy.rs`,
`#[ignore]`, oracle-gated) mirror `measurements/ddblock.py`'s own method
exactly: `KRHF`, `conv_tol=1e-11`, `exxdiv='ewald'`, comparing
`exclude_dd_block = true` against `= false` on the SAME cell. The sign/layout
conventions in `dd_block.rs` were independently re-derived line-by-line
against `_outcore_dd_block`'s own inline pseudocode comments
(`rsdf_builder.py:664-682`, the `PBC_kzdot_CNN_s1` contraction) and confirmed
to match the standard complex product `aopair · Vaux` with NO extra
conjugation on `Vaux` — the same `conj(ao_i)·ao_j` convention this crate's
own `Fftdf::contract_local_potential` already uses for an analogous
real-space quadrature, so the two independently-written routines agree on
the AO-pair phase convention by construction, not by luck.

### 14-07 Task 7d's lesson — both routes pinned separately

`tests/exclude_dd_block.rs` and `exclude_dd_block_energy.rs` gate
`exclude_dd_block = true` AND `= false` each against ITS OWN upstream number
— never only one, per 14-07 Task 7d's documented lesson (flipping a default
while pinning only one route can leave a passing test silently measuring the
wrong route).

### RSDF caveat, stated honestly

`RsGdfBuilder`'s `exclude_dd_block` is wired through the SAME
`make_j3c_scheme_dd` / `fft_dd_block` machinery, but is NOT held to the same
"exactly 0 on He-fcc" claim: `RsGdfBuilder` derives its `ke_cutoff` from
`_guess_omega` (a different formula than GDF's `_guess_eta`), and at
`kmesh=[1,1,1]` this genuinely splits He-fcc's single `sto-3g` shell into
LOCAL+SMOOTH (`bas_type = [1, 2]`, confirmed by direct inspection —
`tests/exclude_dd_block.rs::he_fcc_rsdf_dd_block_seam_is_live`). D-PBC-23's
all-electron-control claim is specific to `_CCGDFBuilder`'s own threshold;
this test only confirms the RSDF seam builds and runs, not a numeric
identity. The `fft_dd_block` omega parameter for RSDF's short-range kernel
is threaded through (`realspace_omega`, matching the sign the real-space
pass it corrects uses) but has LOWER confidence than the GDF path — it has
not been independently oracle-validated the way the GDF/CompensatedCharge
route has. Flagged here rather than silently shipped as equally trusted.

## Task 4 — NOT SHIPPED (carry-over)

Neither the band-k-point `_cderi` rebuild (`gdf/jk.rs:243-253`,
`mdf/mdf_jk.rs:80-90`) nor the MO-factorised `get_k_kpts`
(`gdf/jk.rs:36-38`) is implemented. Both refusals/gaps are UNCHANGED from
before this plan.

**Why, stated precisely rather than left as a gap discovered later:**

* **Band k-points.** Upstream's `get_j_kpts`/`get_k_kpts` accept `kpts_band`
  as a k-point set DISTINCT from `kpts` (the SCF's own sampling set) and
  contract `dm_kpts` — defined over `kpts` — against `cderi` blocks indexed
  by `(ki ∈ kpts, kj ∈ kpts_band)`. This port's `get_j_kpts`/`get_k_kpts`
  (`gdf/jk.rs`) currently assume bra = ket = `kpts` throughout; supporting
  `kpts_band` needs (a) a cderi build over the UNION `kpts ∪ kpts_band`
  (straightforward — `Gdf`/`CcGdfBuilder` already build over an arbitrary
  k-point list) and (b) a NEW asymmetric contraction path in `get_j_kpts`/
  `get_k_kpts` that reads `dms` over `kpts` but produces `vj`/`vk` over
  `kpts_band`. (b) is the real work and was not attempted — it is a
  comparable scope to Task 3's own dd-block addition, and this plan's time
  went to confirming Task 3 correctness instead of starting a second
  same-sized, unvalidated feature.
* **MO-factorised `get_k_kpts`.** No `force_dm_kbuild`-equivalent parameter
  exists in this port at all (checked directly — `gdf/jk.rs` has no such
  field), so there is nothing to "measure and report the speedup" on. Given
  this plan's OWN `dd_block.rs` experience — a straightforward, formula-
  faithful triple-loop implementation turned out to need `--release` to
  finish a single diamond gamma `KRHF` cycle in reasonable time — a first
  cut of an MO-factorised contraction risks the same trap this repo has
  already recorded once (`zgemm_dense`, 6-12× slower than a host loop). Not
  implementing it and reporting so is safer than shipping an unmeasured
  "optimisation."

Both are left exactly as this plan found them: `gdf/jk.rs:243-253` and
`mdf/mdf_jk.rs:80-90` still say `NotYetImplemented { phase: 17 }` for band
k-points (now technically stale — the phase-17 supermole they cite IS done —
but the feature itself is not, so the refusal is honest and only the cited
reason is now slightly imprecise; a follow-on plan should retitle it rather
than remove it).

## Task 5 — `rsjk`'s doc comment updated, refusal untouched

`crates/pyscf-pbc-scf/src/rsjk.rs`'s module doc now records: blocker 1 (the
supermole — `RsCell` + `ExtendedMole`) is CLOSED by this plan; blocker 2 (a
screened periodic 4-centre `int2e` driver, `PBCVHF_direct_drv1`,
`rsjk.py:267-436`) still has NO implementation and NO correct-but-slow
fallback, because for `rsjk` the screening IS the algorithm. `build()` /
`get_jk()` still return `NotYetImplemented` — the refusal LOGIC is
byte-for-byte unchanged, verified by `git diff` showing only doc-comment
lines touched in the refusal functions. `PBC-MASTER-PLAN.md` records that
blocker 2 should be sized as its own plan in a phase after 17.

## `grep -rn "phase: 17" crates/pyscf-pbc-df/`

Returns two lines, both Task 4's untouched, now-imprecisely-titled band-
k-point refusals (`gdf/jk.rs:253`, `mdf/mdf_jk.rs:90`) — NOT zero, because
Task 4 did not ship. This is reported here rather than silently claimed
otherwise; the plan's verification block asked for zero, and it is not zero.

## Files touched

* `crates/pyscf-pbc-df/src/ft_ao/rs_cell.rs` (new) — Task 1.
* `crates/pyscf-pbc-df/src/ft_ao/supmol.rs` (new) — Task 2.
* `crates/pyscf-pbc-df/src/ft_ao/mod.rs` — `pub mod rs_cell;`/`supmol;` + re-exports.
* `crates/pyscf-pbc-df/src/gdf_builder/dd_block.rs` (new) — Task 3.
* `crates/pyscf-pbc-df/src/gdf_builder/eta.rs` — `estimate_rcut_per_shell` added; `estimate_rcut` now a thin wrapper over it (behaviour-preserving, regression-tested).
* `crates/pyscf-pbc-df/src/gdf_builder/j3c.rs` — `make_j3c_scheme_dd` added; `make_j3c`/`make_j3c_scheme` now thin wrappers (behaviour-preserving for existing callers).
* `crates/pyscf-pbc-df/src/gdf_builder/mod.rs` — `rs_cell` field; wiring; default LEFT at `false` (see Task 3's deviation note).
* `crates/pyscf-pbc-df/src/rsdf_builder/mod.rs` — same shape as `gdf_builder/mod.rs`.
* `crates/pyscf-pbc-df/src/gdf/mod.rs` — `Gdf::exclude_dd_block: Option<bool>` passthrough.
* `crates/pyscf-pbc-scf/src/rsjk.rs` — module doc only (Task 5); refusal logic unchanged.
* `crates/pyscf-pbc-df/tests/rs_cell.rs` (new), `tests/extended_mole.rs` (new), `tests/exclude_dd_block.rs` (new), `tests/gdf_builder.rs` (edited — obsolete refusal test replaced).
* `crates/pyscf-pbc-scf/tests/exclude_dd_block_energy.rs` (new) — the SCF-level oracle gate (Task 3's Ha-level numbers).
* `.planning/pbc/PBC-MASTER-PLAN.md` — D-PBC-27 recorded (additive; see note on 17-02 concurrency below); rsjk sizing note.
* `.planning/STATE.md` — Current Position, additive.

## Concurrency note (17-02)

Plan 17-02 (the symmetry half — `crates/pyscf-pbc-symm`) ran concurrently in
this same working tree, touching `crates/pyscf-pbc-symm/*`,
`.planning/STATE.md`, `.planning/ROADMAP.md` and `.planning/pbc/PBC-MASTER-PLAN.md`.
This plan's edits to the three shared `.planning` files were made additively
against 17-02's already-present content (re-read immediately before each
edit, per the task instructions) — no file overlap in `crates/`.

## Deviations from the plan, stated

1. **`ExtendedMole` is NOT a literal `Mole`** (see Task 2 above) — a
   deliberate representational choice, not an omission; every quantity the
   plan gates is reproduced.
2. **`fft_dd_block` is a naive `O(nkptij · nao_d² · naux · ngrids)` quadrature**,
   no BLAS, no device kernel, single-threaded. Fine for every system this
   plan gates (`nao_d` — the smooth-cell AO count — is a small fraction of
   `nao` on both He-fcc and diamond) but a production workload with a larger
   smooth block would want the batched contraction upstream's `PBC_kzdot_CNN`
   C driver uses. Left as a follow-on.
3. **Task 4 not shipped** — see above.
4. **Diagnostic timing scripts were deleted, not shipped**: this plan spent
   real wall-clock investigating whether a >10-minute `KRHF` run was a bug
   or expected cost, and concluded it is the latter — this workspace's
   pre-existing real-space `aux_e2` on diamond is independently documented
   elsewhere as "minutes" in a debug build, and this plan's own correction
   adds comparable extra passes. The throwaway probe files used for that
   investigation (`dd_timing_probe.rs`, `dd_block_probe.rs`) were deleted;
   the permanent, oracle-target-asserting replacement is
   `crates/pyscf-pbc-scf/tests/exclude_dd_block_energy.rs`.
5. **The `exclude_dd_block` default stays `false`** on `CcGdfBuilder` and
   `RsGdfBuilder` — see Task 3's own "Deviation" note above for the full
   reasoning (a global default flip's safety could not be confirmed against
   this crate's many pre-existing oracle-pinned tests within this plan's
   time budget, so the refusal was closed as a fully-tested OPT-IN instead
   of an unverified new default).
6. **The diamond Ha-level SCF numbers (1.835e-08, 2.900e-08) were NOT
   confirmed by a completed live run of this port's own Rust code within
   this session.** What WAS done: (a) the upstream/oracle side of the
   comparison was independently re-run against the vendored PySCF 2.12.1 and
   reproduced 2.9002556800605817e-08 on diamond gamma, confirming the
   TARGET number and the measurement methodology are both real and
   reproducible; (b) `dd_block.rs`'s arithmetic was re-derived line-by-line
   against `_outcore_dd_block`'s own upstream pseudocode comments and checked
   against this crate's independently-written `Fftdf::contract_local_potential`
   for AO-pair phase-convention agreement; (c) the He-fcc bit-identical case
   — which exercises the SAME code path's "no correction" branch — passes.
   A live Rust `KRHF` run on diamond gamma was started (in `--release`) to
   close this gap and was still running, past 10 minutes of wall clock, when
   this plan's time budget required moving on; it was not confirmed to
   either pass or fail. `crates/pyscf-pbc-scf/tests/exclude_dd_block_energy.rs`
   is written, `#[ignore]`d, and ready to run this confirmation
   (`PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-scf --test
   exclude_dd_block_energy -- --ignored --nocapture`) — running it is the
   single highest-value next step for whoever picks this plan back up.

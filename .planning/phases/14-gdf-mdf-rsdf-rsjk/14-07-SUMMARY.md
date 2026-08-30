# 14-07 — RSDF. **7a shipped. 7b / 7c / 7d BLOCKED on a cintx capability that does not exist.**

**Status:** sub-task **7a complete and green** — 10 tests in
`crates/pyscf-pbc-df/tests/rsdf_builder.rs`, every number gated against
`measurements/omega.out` at 1e-12. Sub-tasks 7b, 7c and 7d are blocked and the
blocker is recorded here, in the module docs, and in a test.

Plan 14-07 sequenced 7a first for exactly this reason: *"Do 7a completely, with
its tests green, before writing a line of 7b."* It also named the failure mode
in advance, in Task 7b's own words:

> **Check first that the cintx resolver exposes a short-range `int3c2e`**; if it
> does not, this is the plan's one real blocker and it must be reported as such,
> not worked around with a numerically different kernel.

## The blocker, with the evidence

`_RSGDFBuilder.get_2c2e` needs a short-range `int2c2e` and its `outcore_auxe2` a
short-range `int3c2e`. Range separation is **not** a distinct integral symbol —
upstream never calls an `int2e_sr_*` — it is libcint's `PTR_RANGE_OMEGA`
(`env[8]`) toggle around the standard symbol. cintx cannot be asked to set it:

1. **`cintx_runtime::ExecutionOptions`** (`cintx/crates/cintx-runtime/src/options.rs:96`)
   carries `f12_zeta` (`env[9]`), `rinv_orig` (`env[4..6]`) and `common_orig`
   (`env[1..3]`). There is **no `range_omega` field**.
2. **No kernel reads `env[8]`.** `cintx-cubecl/src/kernels/center_3c2e.rs` and
   `two_electron.rs` do not mention `omega` at all.
   `cintx-compat/src/raw.rs:35-41` names `PTR_RANGE_OMEGA = 8` only in a warning
   not to overwrite the slot.
3. **The resolver knows `int3c2e` and `int3c2e_ip1`** and no range-separated
   variant — correctly, per (1).
4. **The gap is already on this repository's record.**
   `crates/pyscf-gto/src/range_coulomb.rs` documents it as Phase 4's Open
   Question A5 / cintx#11: *"cintx reads `env[8]`, but its safe API … exposes
   only `f12_zeta` (env[9]) … there is no `range_omega` (env[8]) setter on the
   safe path."* Phase 4 shipped the set/restore semantics and CI-gated the
   numerical RSH assertion behind the same gap.
5. **A second, independent obstruction on this path.**
   `incore::aux_e2` reaches cintx through
   `pyscf_gto::build_image_expanded_with_aux`, which builds its `BasisSet` from
   `cell.mol._atom` / `_basis` — the per-element parsed basis — and not from an
   `_env` array. So even `range_coulomb.rs`'s own workaround (write
   `mol._env[8]` directly around a standard `intor` call) is not reachable from
   the periodic 3-centre driver. Closing (1) would still leave this.

Substituting the full-range kernel would produce a builder that runs, converges,
and is silently a different method. `RsGdfBuilder::build`, `Rsdf::build`,
`density_fit(DfKind::Rsdf)` and `RangeSeparatedJkBuilder::build`/`get_jk` all
return `NotYetImplemented { phase: 14 }` naming the gap (D-PBC-20), and
`rs_gdf_builder_refuses_and_names_the_cintx_gap` asserts the message text so the
next reader does not re-derive any of this.

## Consequences

* **Gate 3 is unreachable this phase.** It compares `|E_KRHF(GDF) − E_KRHF(RSDF)|`
  against upstream's own floor (1.353e-08 diamond 2×2×2, 4.566e-09 gamma,
  1.113e-10 He-fcc). One of the two builders does not exist.
* **Task 7d does not happen.** `Gdf::prefer_ccdf` stays `true`; the committed
  `df_swap` baseline does not move.
  `gdf_default_route_has_not_flipped` pins that.
* `_RSMDFBuilder` (14-06's other half) is blocked for the same reason, which is
  why `measurements/mdfladder.out` — recorded on the RS route — could not gate
  14-06 and `mdfladder_cc.out` had to be added.

## 7a — what DID ship, and why it is worth shipping alone

`crates/pyscf-pbc-df/src/rsdf_builder/omega.rs`: `OMEGA_MIN`,
`RCUT_THRESHOLD`, `guess_omega`, `estimate_omega_min`,
`estimate_ke_cutoff_for_omega`, `estimate_omega_for_ke_cutoff`,
`estimate_rcut`, `estimate_ft_rcut`, `estimate_rs_2c2e_rcut`, `estimate_meshz`,
`round_off_to_odd_mesh`, `gaussian_int`, `weighted_coulg_lr`,
`weighted_coulg_sr`, plus `estimate_ke_cutoff_pgto_4c`.

Three separate consumers need these regardless of the blocker: `rsjk` (14-08,
which imports them rather than re-porting), RSH functionals (`JkOpts::omega`,
already threaded through `get_coulG`), and Phase 17.

### Task 0 — `measurements/omega.py` / `.out`, recorded before a line was written

| quantity | He-fcc 2×2×2 | diamond 2×2×2 |
|---|---|---|
| `_guess_omega → omega` | **0.739358637866536** | **0.601955030338906** |
| `_guess_omega → mesh` | `[11,11,11]` | `[11,11,11]` |
| `_guess_omega → ke_cutoff` | 30.7085675919949 | 21.7218834404379 |
| `estimate_omega_min` | 0.324467042356544 | 0.180615163949558 |
| `estimate_ke_cutoff_for_omega(OMEGA_MIN)` | 0.335874219073685 | 0.330921403072804 |
| `estimate_ke_cutoff_for_omega(omega)` | 24.2472773849789 | 16.8733499340885 |
| `estimate_omega_for_ke_cutoff(20)` | 0.596678472583996 | 0.578002439380595 |
| `_estimate_meshz` | 43 | 47 |
| `estimate_rs_2c2e_rcut(omega)` | 9.26629761172378 | 16.41536798946 |
| `estimate_rcut` (max) | 11.130814509359459 | 18.568121481440436 |
| `estimate_ft_rcut` (max) | 12.274614504009085 | 20.481090224560270 |

All reproduced to 1e-12 (1e-11 on the radii, which are `log`/`sqrt` compositions
of ~20-Bohr quantities). `_guess_omega`'s gamma values —
`omega = 1.03510209301315`, `mesh = [15,15,15]` on He-fcc;
`omega = 0.720159089892724`, `mesh = [13,13,13]` on diamond — are gated too.

### The identity that catches an erf/erfc swap

`weighted_coulG_SR + weighted_coulG_LR == weighted_coulG` at **every** `G`,
residual **exactly 0** — upstream's is 0 too, because `weighted_coulG_LR` is
*defined* as the difference (`rsdf_builder.py:195-200`, and its comment explains
that a direct `+omega` evaluation would be wrong whenever the CELL carries an
omega). A sum alone cannot distinguish the two halves, so the test also pins
their shapes: `sr[1] > lr[1]` at the smallest non-zero `G` (the short-range half
must dominate there), and both vanish at `G = 0`. The first four values of each
half are pinned to upstream at 1e-14.

## Three things that look trivial in this file and are not

* **`estimate_rs_2c2e_rcut` reads `auxcell.rcut`**, and that is the rcut of the
  MODRHO-normalised auxiliary cell. Using a plain `make_auxcell` instead moves
  it by 2.13e-02 — the modrho rewrite rescales the `_env` contraction
  coefficients and therefore the cell's own cutoff. The test says so.
* **`estimate_ke_cutoff_for_omega` uses `aft._estimate_ke_cutoff`**, the
  4-centre Coulomb-repulsion form, NOT `cell._estimate_ke_cutoff`
  (`pyscf_pbc_gto::cutoff::estimate_ke_cutoff_pgto`), which is the
  nuclear-attraction one. They differ in `norm_ang` (squared), in the power of
  `2α`, and in the iteration multiplier. It also **discards** the extracted
  contraction coefficients and substitutes `gto_norm(l, α)`. Phase 13 shipped a
  defect from exactly this confusion (21.186 against 20.420 Bohr).
* **`estimate_ft_rcut`'s two `r0` iterations are different expressions** —
  `fl = 2πr₀/θ + 1` then `fl = 2π/vol · r₀/θ`. Copying either one twice changes
  the answer.

## Deviation

`estimate_rcut` and `estimate_ft_rcut` take a `_RangeSeparatedCell` upstream;
this port has none (D-PBC-21/23) and calls them with the plain cell.
`omega.out` records both, and the **maxima agree exactly** — the split only
refines the smaller radii (diamond: `[13.78, 17.73, 14.18, 18.57, …]` split
against `[17.73, 18.57, 17.73, 18.57]` plain), so the plain-cell call is the
conservative one.

---

# Sub-tasks 7b + 7c (partial) — `_RSGDFBuilder` — 2026-08-30

**Status: 7b DONE. 7c PARTIAL (`_RSNucBuilder` and `rsdf_helper`'s prescreen not
ported). 7d NOT DONE (deliberately).** Gate 3 is MET; see
`14-VERIFICATION.md` §5.

## What unblocked this

Task 7b's own instruction was to check whether cintx exposed a short-range
`int3c2e` and, if not, to **report** rather than substitute a different kernel.
Phase 14 closed on that report. **D-PBC-24 then supplied the capability** —
`ExecutionOptions::range_omega`, part of the workspace query because short range
doubles the Rys roots. The second obstruction 14-VERIFICATION recorded (that
`aux_e2` builds its `BasisSet` with no `_env`, so `range_coulomb.rs`'s
`OmegaGuard` had nothing to write into) **turned out never to matter: ω rides in
the OPTIONS, not in the basis.**

## What was written

| file | what |
|---|---|
| `gdf_builder/fuse.rs` | `unfused_auxcell` — a degenerate `FusedCell` (`fused == auxcell`, no model charges) so the RS route reuses one driver instead of forking it |
| `rsdf_builder/j2c.rs` | `get_2c2e` (SR analytic + LR reciprocal), `weighted_ft_ao` (the LR remainder on every aux column), `rs_vbar`, `weighted_coulg_at` |
| `rsdf_builder/mod.rs` | `RsGdfBuilder::{build, make_j3c}`, `j2c_mesh` |
| `gdf_builder/j3c.rs` | `Scheme::RangeSeparated { omega }` — SR real-space pass, LR reciprocal pass with the sign FLIPPED, `naux` rows, Cholesky-first solve |
| `gdf/mod.rs` | `prefer_ccdf = false` now routes to the RS builder instead of refusing; `rs_rcut` / `rs_mesh` overrides |
| `rsdf.rs` | `Rsdf` IS a `Gdf` with `prefer_ccdf = false`, as `RSGDF` subclasses `GDF` upstream |
| `density_fit.rs` | `DfKind::Rsdf` builds instead of refusing |

`add_ft_j3c` is now one kernel with a `sign` parameter: CC and MDF **remove** a
projection (`-1`), RS **adds** the long-range remainder (`+1`,
`rsdf_builder.py:806-811`, where all four `lib.ddot`s carry `+1`). Same two
products either way — which is why it stayed one function.

## Result

`conv_tol = 1e-12`, vs vendored PySCF 2.12.1.

He-fcc `sto-3g` 2×2×2 (all-electron):

| route | upstream | port | error |
|---|---|---|---|
| RSDF | −2.80842508717097 | −2.80842508693849 | **2.325e-10** |
| GDF (CC) | −2.80842508664874 | −2.80842508692377 | **2.750e-10** |

Diamond `gth-szv`/`gth-pade` gamma (pseudopotential):

| route | upstream | port | error |
|---|---|---|---|
| RSDF | −10.14369692267123 | −10.14369690652303 | **1.615e-8** |
| GDF (CC) | −10.14369242019033 | −10.14369244092593 | **2.074e-8** |

**The original Gate 3 criterion is MET on diamond**: the port's `|CC − RS|` is
4.465597e-6 against upstream's 4.502481e-6 — ratio **0.9918**. On He-fcc the same
ratio is 0.028 and does not discriminate, because upstream's two routes differ
there almost entirely through the two splits this port has in neither route.
See `14-VERIFICATION.md` §5.

`_RSNucBuilder`'s absence does not show at 1e-8 even on the pseudopotential cell:
RSDF's diamond error is smaller than GDF's.

## Two findings worth keeping

**1. `estimate_rs_2c2e_rcut` is load-bearing, not a tuning knob.**
`auxcell.rcut` is an ORBITAL radius; the metric's lattice sum is over a
two-centre COULOMB interaction `erfc(ωR)/R`, which reaches much further (at
ω = 0.42, precision 1e-8: ~9.6 Bohr from the erfc alone). Upstream sets it at
`rsdf_builder.py:274`. Without it the real-space SR metric was **1.25e-4** off
its reciprocal equivalent at every k-point, worth **8.57e-5 Ha**. With it, the
metric matches a converged reciprocal reference to **2e-12**.

**2. `_guess_omega` takes the ORBITAL cell here, where upstream passes the
auxcell** — the priced cost of having no `_RangeSeparatedCell`. Upstream's
`exclude_d_aux`/`exclude_dd_block` route what a coarse grid cannot resolve
around the grid; this port resolves it instead. At upstream's `[7,7,7]` the
error is **8.670e-7**; at this port's `[11,11,11]`, **1.97e-10**. That
`[11,11,11]` is exactly what `measurements/omega.out` already recorded.

## A trap for the next reader

The reciprocal `Σ_G conj(auxG) coulG_SR auxG` converges **slowly** — the SR
kernel is not smooth in `G` — so it is a bad reference unless converged. Meshes
21/41/61/81 give 1.02e-1 / 7.64e-5 / 2.02e-9 / 1.38e-12 against the analytic SR
sum. A mesh-41 reference made a correct metric look 7.6e-5 wrong.

## Not done, and why

* **`_RSNucBuilder`** (`rsdf_builder.py:1098-1311`). `Gdf` serves
  `get_nuc`/`get_pp` from the compensated route for both schemes. **Measured not
  to matter at 1e-8**: on diamond gamma — a pseudopotential cell, where it would
  show — RSDF's error against upstream is 1.615e-8, *smaller* than GDF's
  2.074e-8. Still a fidelity gap worth closing, no longer a suspected accuracy
  one.
* **`rsdf_helper.py`'s prescreen.** Its absence keeps MORE primitives than
  upstream — conservative, as 14-05 was toward `ExtendedMole.strip_basis`.
* **Task 7d, the `prefer_ccdf` flip.** It moves a committed reference energy
  (diamond 2×2×2, a documented 5.960e-07 step) and Task 7d requires that be its
  own cited edit rather than a side effect of shipping the builder.
  `tests/rsdf_builder.rs::gdf_default_route_has_not_flipped_yet` holds the line.

---

# `_RSMDFBuilder` + Task 7d — 2026-08-30

## `_RSMDFBuilder` (`mdf.py:238-353`)

Upstream is a subclass of `_RSGDFBuilder` overriding three methods; this port
is a `mixed: bool` on the one builder, for the reason 14-02 made `Scheme` a tag.
What it changes:

| | `_RSGDFBuilder` | `_RSMDFBuilder` |
|---|---|---|
| reciprocal weight | `coulG − coulG_SR` | `−coulG_SR` |
| metric | `SR_analytic + FT_full − FT_SR` | `SR_analytic − FT_SR` |
| metric mesh | tightened `j2c_mesh` | the builder's own `mesh` |
| solve | Cholesky first | eigen always |
| every kernel | plain | + MDF's `±Gmax±0.5` edge screen |

**Measured at matched meshes** vs upstream's `df.MDF()` default, He-fcc 2×2×2:
**3.209e-10** (11), **1.897e-11** (15), **7.808e-12** (21).

### Two bugs, both caught by the oracle

1. **The edge screen reaches the SHORT-range kernel.** `_RSMDFBuilder` ends with
   `weighted_coulG = MDF.weighted_coulG` (`mdf.py:353`), and
   `weighted_coulG_SR` is defined in terms of it — so MDF's `±Gmax ± 0.5`
   screen applies to the SR kernel too. It fires at a half-integer scaled
   k-point, i.e. at EVERY k-difference on a 2×2×2 mesh. Omitting it: **1.176e-4 Ha**.
2. **The rcut precision differs between the two files** — `mdf.py:265` passes
   none, `rsdf_builder.py:274` passes `precision**1.5`. Matching upstream's
   looser value measured **worse** (1.324e-6 vs 1.160e-6): a smaller precision
   gives a LARGER radius, and this radius feeds a real-space sum whose
   truncation no mesh can compensate. Both schemes keep the tighter one.

### The mesh means something different for MDF

For GDF the mesh only decides how accurately the long-range half is evaluated,
so finer is strictly closer to the exact GDF answer. **For MDF the plane-wave
set is part of the basis** (`<g|g> − <g|G><G|g>`, with `aft_jk` adding the
residual back over the same `{G}`), so two meshes are two different — equally
valid — MDF approximations. An MDF energy is only comparable against another at
the SAME mesh, which is why the gate forces it on both sides. The port's default
(`[11,11,11]`, from the cell) sits 1.160e-6 from upstream's (`[7,7,7]`, from the
auxcell); the ladder above is what shows that to be grid, not algebra.

## Task 7d — the flip

`Gdf::prefer_ccdf` now defaults to `false`, matching upstream.

`df_swap.rs::krhf_on_gdf_matches_upstream_he_fcc` pins **both** routes against
their own upstream numbers. That is the substance of the task, not ceremony: the
two routes differ by 5.222e-10 on He-fcc, which is *inside* that test's 1e-9
bar, so the pre-flip version would have kept passing while silently measuring
the other route. `rsdf_builder.rs::gdf_default_route_is_range_separated` pins the
default in the new direction.

The flipped default is also the faster one — 1.3 s against 6.6 s on He-fcc
2×2×2, because the short-range real-space sum is cheaper than the compensated
one.

**A caveat I had to withdraw.** I first recorded the flipped default as a
hybrid — "RS fitting, CC nuclear" — matching neither upstream route. That is
wrong: this port uses NEITHER split nuclear builder. `gdf::nuc::get_nuc` goes
straight to AFTDF at the cell's converged mesh, oracle-gated at 2.755e-12, which
is strictly more accurate than either `_CCNucBuilder` or `_RSNucBuilder`. Both
of those are *performance* devices that let the nuclear part run on a tiny mesh,
and 14-04 measured that using that mesh WITHOUT the split costs 0.0743 Ha. So
`_RSNucBuilder` is the same performance carry-over 14-03 opened, and the flip
does not widen anything. The measurement agrees: on diamond gamma the RS route's
error (1.615e-8) is smaller than the CC route's (2.074e-8).

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

# 20-05 SUMMARY — `GDF.get_jk(omega)` / `MDF.get_jk(omega)`: refusals replaced, all four routes under 1.353e-08

**Executed:** 2026-09-14. All four tasks done. The general `omega` refusal is gone from both
`gdf/jk.rs` and `mdf/mdf_jk.rs`. Two upstream sub-branches this port does not have stay as **named**
refusals (listed below). The short-range route first failed its gate at **1.4e-4**. That traced to
**two defects in shared code**, which were fixed rather than absorbed (deviations 2 and 3). No git
operations were run (D-20-A). The work was done in an isolated copy (`../pyscf_rs-p2005`, now
deleted), and only the nine files listed at the end were copied back. Each was checked unchanged in
the main tree since the snapshot.

## What upstream 2.12.1 actually does (it differs from the plan)

`df.py:459-479` (GDF) and `mdf.py:180-199` (MDF):

| request | upstream | port (this plan) |
|---|---|---|
| `omega > 0`, and GDF: `dimension >= 2 and low_dim_ft_type != 'inf_vacuum'` / MDF: `... or ...` (kept as written) | `mydf = aft.AFTDF(cell, self.kpts)`, `mesh = cell.cutoff_to_mesh(estimate_ke_cutoff_for_omega(cell, omega))`, `AFTDF.range_coulomb` sets `cell.omega` (`aft.py:552-582`) | `gdf::jk::rsh_aftdf` → `aft_jk::get_jk` with the omega **in the cell** (`_env[PTR_RANGE_OMEGA]`) and `JkOpts.omega = None` |
| otherwise | `mydf = self`; `range_coulomb` asserts `omega < 0` for `dimension != 0` (`df.py:520-521`), yields `self.copy().reset()` with `cell.omega = auxcell.omega = omega`, cached in `_rsh_df['%.6f' % omega]`; its build runs `_RSGDFBuilder` / `_RSMDFBuilder` with `self.omega = -cell.omega` (`rsdf_builder.py:83-90`) | new `Gdf::range_coulomb` / `Mdf::range_coulomb` (same key and cache; MDF copy keeps the parent's mesh as upstream's `copy()` does). `RsGdfBuilder::new` now seeds `omega = -cell.omega` for a negative `cell.omega`, and `build()` refuses a positive one (`:89-90` raises) |
| GDF `omega == 0` | plain request (`omega != 0` test) | plain request |
| MDF `omega == 0` | `AssertionError` | error |

**Deviation 1 (fact, from the plan's `<objective>`).** The plan routed the GDF case "through
`RsGdfBuilder` with the requested omega". But upstream GDF *also* swaps to AFTDF for `omega > 0`,
the long-range case that CAM-B3LYP uses. Only `omega < 0` reaches the fitted builder, and HSE's
`get_k(omega=-omega)` (`pbc/dft/krks.py:235`) is what sends it there. Both branches were ported, and
both are gated.

The omega is carried in the **cell**, never as an explicit kwarg. The reason is
`pbc.py:480-484`: an explicit `omega` switches the Ewald probe to full range, while `cell.omega`
keeps it attenuated. Upstream's RSH branch uses `cell.omega`, so `exxdiv='ewald'` matches only this
way.

Still refused (`NotYetImplemented { phase: 20 }`). These are upstream routes the port does not have.
None of them returns plain Coulomb:
* `omega < 0` with `prefer_ccdf = true`. Upstream would run `_CCGDFBuilder` / `_CCMDFBuilder` on the
  attenuated cell (`df.py:304`, `mdf.py:126`). **Note:** the port's `Mdf::new` default is
  `prefer_ccdf = true`, which differs from upstream's `False`. A default-constructed port `Mdf`
  therefore refuses `omega < 0`. Set `prefer_ccdf = false` to get upstream's route.
* `omega > 0` on a 0-D cell (GDF), or on a 0-D `inf_vacuum` cell (MDF): `_CC*Builder` on a
  long-range cell.

## The gate — `crates/pyscf-pbc-df/tests/gdf_omega.rs` (new, 4 tests, tag `T2`)

He-fcc `sto-3g` 2×2×2, `omega = ±0.33`, model density handed to upstream as literals. The
comparison is the element-wise worst deviation over every k-point, for `exxdiv=None` and
`exxdiv='ewald'`, against **upstream's own RSH answer** (never against plain GDF/MDF). The floor is
**1.353e-08** (`measurements/README.md` §1 row 5). `pyscf==2.12.1` is asserted, and when
`oracle_python()` returns `None` the test skips.

| test | route | exxdiv | \|dvj\| | \|dvk\| | verdict |
|---|---|---|---|---|---|
| `gdf_lr_matches_upstream` | GDF → AFTDF [7,7,7] | None / ewald | 8.034e-17 / 8.034e-17 | 4.564e-13 / 2.223e-11 | PASS |
| `gdf_sr_matches_upstream` | GDF `range_coulomb` → RSGDF | None / ewald | 1.939e-9 / 1.939e-9 | 2.823e-10 / 2.823e-10 | PASS |
| `mdf_lr_matches_upstream` | MDF → AFTDF [7,7,7] | None / ewald | 8.034e-17 / 8.034e-17 | 4.564e-13 / 2.223e-11 | PASS |
| `mdf_sr_matches_upstream` | MDF `range_coulomb` → RSMDF [7,7,7] | None / ewald | 1.830e-9 / 1.830e-9 | 5.130e-10 / 5.130e-10 | PASS |

Runtime: 45.9 s for the four tests with `--test-threads 1` (78.7 s in an earlier loaded run), so
**T2**. 20-03 has not executed yet and owns the nightly workflow list, so the test was **not**
added to `.github/workflows/nightly-cross-crate.yml` by this plan. 20-03 should add `gdf_omega` to
it.

## Deviation 2 — the short-range 3-centre tensor was truncated (shared code)

The first SR run gave GDF **|dvj| 1.414e-4 / |dvk| 1.126e-5** and MDF **5.243e-3 / 8.006e-3**.
The trace went like this:
* The fitted SR `(kk|k'k')` was 0.36812783 in the port and 0.36824252 upstream. The exact AFT
  value has `full − SR = 4.666e-5`. Upstream's fit reproduces that (4.667e-5); the port's gave
  1.61e-4. **The port was wrong, not upstream.**
* The SR metric `j2c` agreed with upstream to 2.5e-14, so the fault was not there.
* The real-space `(ij|P)` was off by up to **6.674e-5**, and it did not change at all when `rcut`
  went from the estimate to 40 Bohr.

Cause: `incore::int3c::aux_e2_intor` screens with an **overlap** radius (`r_s + r_P`) and an
overlap-shaped Gaussian prescreen. Its own module docs say that is exact only for the neutralised
fused cell. The range-separated route passes charged, unfused auxiliary functions under
`erfc(|ω|r)/r`, and those interact out to the kernel's range. The fix applies to `omega < 0` only.
The neighbour list widens to the caller's SR `rcut`, which is upstream's `strip_basis(rcut_sr)`
radius. A new `prescreen_exponent_sr` uses `theta = 1/(1/(a+b) + 1/c + 1/ω²)`, the `theta` from
`rsdf_builder.py`'s `estimate_rcut`. After the fix, real-space `(ij|P)` is **9.8e-10** off upstream
and GDF SR passes. The `omega = None` path (CC/MDF fused) is unchanged.

## Deviation 3 — RSMDF metric radius precision now mirrors `mdf.py:264`

After deviation 2, MDF SR measured **|dvj| 2.155e-8** (FAIL). The gap shrank with mesh (6.2e-9 at
[11]³, 4.0e-11 at [15]³). Running *upstream* with RSGDF's `precision**1.5` in its RSMDF metric
radius dropped the gap to **1.791e-9**. So the cause was `rsdf_builder/j2c.rs` applying RSGDF's
`auxcell.precision**1.5` to both schemes. The comment there had measured upstream's looser value as
worse, but that measurement was taken while the deviation-2 truncation was live. The mixed scheme
now uses `auxcell.precision`, as `mdf.py:264` does: MDF SR **1.830e-9**, PASS.

## Blast radius — pre-existing gates re-run after both fixes (all exit 0)

| gate | before (recorded) | after |
|---|---|---|
| `pyscf-pbc-scf gate3_rsdf` RSDF He-fcc KRHF | 2.325e-10 | **2.396e-10** (< 1e-9) |
| same, GDF(CC) | 2.750e-10 | 2.750102e-10 (unchanged) |
| RSMDF mesh 11 / 15 / 21 | 3.049e-10 / 1.900e-11 / 7.809e-12 | **3.211e-10 / 1.897e-11 / 7.806e-12** (< 1e-9) |
| RSMDF CC control | 2.827e-10 | 2.730e-10 |
| `band_kpoints` GDF band | ~1.394e-9 | **1.4116e-9** \|dvj\|, 1.889e-11 \|dvk\| (< 2e-9); MDF band ok |
| `df_swap krhf_on_gdf_*` (2) | pass | pass |
| `gdf_builder helium_fused_j3c_and_j2c_match_upstream` | pass | pass |
| `incore` isolated-cell oracles (2) | pass | pass |

`df_jk_gdf.rs::omega_is_refused` asserted the old refusal, so it was **restated** as
`omega_is_honoured_not_ignored`. It runs offline and checks two things. First, `vj`/`vk` at
`omega = ±1.0` differ from full range by more than 1e-3. At `±0.2` the SR matrix of this dense
cell sits legitimately within 8.4e-10 of full range, so a small omega cannot tell a correct answer
from an ignored one. Second, `prefer_ccdf = true` with `omega < 0` must still refuse, naming
`_CCGDFBuilder`.

Not re-run: `gdf_builder`'s diamond fused-cell `--ignored` test. It uses the CC route with
`omega = None`, which is untouched, and it was killed after 27 min (D-20-D: T3, outcome not
observed). `gate3_rsdf` diamond gamma was also not run (T3).

## Verification (commands run in the isolated copy; identical sources copied back)

```bash
export PYSCF_ORACLE_VENV=1 CARGO_TARGET_DIR=$PWD/target/gate CARGO_PROFILE_RELEASE_LTO=false
cargo test --release -p pyscf-pbc-df --test gdf_omega --test df_jk_gdf -j 4 -- --ignored --nocapture --test-threads 1
#   test result: ok. 4 passed (gdf_omega) — the table above; EXIT=0
cargo test --release -p pyscf-pbc-df --test df_jk_gdf -j 4          # 6 passed, EXIT=0
cargo test --release --no-fail-fast -p pyscf-pbc-df --test gdf_omega --test gdf_builder \
  --test band_kpoints --test df_jk_gdf --test rsdf_builder --test gdf --test mdf \
  --test exclude_dd_block --test incore --test rsdf -j 4
#   0+6+2+12+16+0+11+5+5+10 passed, 0 failed; EXIT_A=0 (14m29s)
cargo test --release --no-fail-fast -p pyscf-pbc-df --test band_kpoints --test incore -j 4 -- --ignored
#   2 + 2 passed; EXIT_B=0
cargo test --release -p pyscf-pbc-df --test gdf_builder -j 4 -- --ignored --exact helium_fused_j3c_and_j2c_match_upstream
#   1 passed; EXIT_B2=0
cargo test --release --no-fail-fast -p pyscf-pbc-scf --test gate3_rsdf -j 4 -- --ignored --nocapture he_fcc
#   2 passed (339.5 s); EXIT_C=0
cargo test --release -p pyscf-pbc-scf --test df_swap -j 4 -- krhf_on_gdf     # 2 passed; EXIT_C2=0
grep -n 'NotYetImplemented' crates/pyscf-pbc-df/src/gdf/jk.rs crates/pyscf-pbc-df/src/mdf/mdf_jk.rs
#   only the two named attenuated-_CC*Builder / 0-D sub-branch refusals (gdf/jk.rs:820,834;
#   mdf/mdf_jk.rs:160,170) + doc lines — no general omega refusal
```

`rustfmt --edition 2024` was run on the touched files only. Rustfmt follows `mod` declarations and
reformatted the untouched `gdf/cderi_store.rs` in the copy; that file was **not** copied back. No
CubeCL kernel code was written, and no build errors hit the cubecl protocol.

## Files changed (copied back into the main tree)

- `crates/pyscf-pbc-df/src/gdf/jk.rs` — `get_jk` RSH dispatch, `get_jk_rsh`, `cell_with_omega`, `rsh_aftdf`
- `crates/pyscf-pbc-df/src/gdf/mod.rs` — `Gdf::range_coulomb` + `_rsh_df` cache
- `crates/pyscf-pbc-df/src/mdf/mdf_jk.rs` — `get_jk` RSH dispatch, `get_jk_rsh`
- `crates/pyscf-pbc-df/src/mdf/mod.rs` — `Mdf::range_coulomb` + cache
- `crates/pyscf-pbc-df/src/rsdf_builder/mod.rs` — `new` seeds `omega = -cell.omega`; `build` refuses `cell.omega > 0`
- `crates/pyscf-pbc-df/src/rsdf_builder/j2c.rs` — mixed-scheme metric radius precision (deviation 3)
- `crates/pyscf-pbc-df/src/incore/int3c.rs` — SR neighbour list + `prescreen_exponent_sr` (deviation 2)
- `crates/pyscf-pbc-df/tests/gdf_omega.rs` — new gate
- `crates/pyscf-pbc-df/tests/df_jk_gdf.rs` — `omega_is_refused` → `omega_is_honoured_not_ignored`

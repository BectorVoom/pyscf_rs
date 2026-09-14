# 20-04 — GDF `KRKS` k-symmetry vs full BZ: bisect by route step

**Measured:** 2026-09-14, working tree on HEAD `0f8b58d` + uncommitted Phase-18
work + the D-20-E `fftdf.rs` re-key (the FFTDF control below was measured AFTER
that fix; the GDF arms never touch the `coulG` cache).
**Harness:** `crates/pyscf-pbc-dft/tests/gdf_ksymm_bisect.rs` (new file, three
`#[ignore = "T3: 20-04 bisect diagnostic ..."]` tests). **No production `src/`
file was edited by this plan.**
**Build/run shape:** `CARGO_TARGET_DIR=$PWD/target/gate
CARGO_PROFILE_RELEASE_LTO=false cargo test --release -p pyscf-pbc-dft --test
gdf_ksymm_bisect -j 4 -- --ignored --nocapture --test-threads=1 <name>` under
`systemd-run --user --scope -p MemoryMax=16G`. Raw logs:
`target/gdf_ksymm_bisect_t0.log`, `target/gdf_ksymm_bisect_t1.log` (exit=0,
3175 s), `target/gdf_ksymm_bisect_t2.raw`.

## Verdict

**The first step exceeding 1e-9 is the XC quadrature (step 6), and the cause is
a DEFECT — class (a) — in the full-BZ REFERENCE arm, not in GDF and not in the
k-symmetry layer.**

`Krks::from_df` builds its uniform XC grid on `with_df.mesh()`:

```rust
// crates/pyscf-pbc-dft/src/krks.rs:93
let grids = PeriodicGrids::uniform(with_df.cell(), Some(with_df.mesh()))?;
```

For `Gdf`, `mesh()` is the range-separated builder's LONG-RANGE Coulomb mesh
(`crates/pyscf-pbc-df/src/gdf/mod.rs:307-323` when measured; `:363-380`
after another agent's concurrent 09:05 edit to that file, body unchanged —
both binaries were built before that edit), which on `si [2,2,2]` is
**`[13,13,13]` (2,197 points)**. `KsymAdaptedKrks` — both `::new`
(`krks_ksymm.rs:112`) and the gate's explicit `from_df(..., grids)` — uses
**`cell.mesh = [35,35,35]` (42,875 points)**. Upstream builds the grid from the
cell in both classes: `rks.py:272` `self.grids = gen_grid.UniformGrids(self.cell)`,
`gen_grid.py:72` `self.mesh = cell.mesh`.

| | expected (upstream) | actual (port) |
|---|---|---|
| full-BZ `KRKS` over `GDF`, XC grid | `cell.mesh` = `[35,35,35]` | `Gdf::mesh()` = `[13,13,13]` |
| full-BZ `KRKS` over `FFTDF`, XC grid | `cell.mesh` | `Fftdf::mesh()` = `[35,35,35]` (coincides) |
| `KsymAdaptedKrks` over either DF, XC grid | `cell.mesh` | `cell.mesh` = `[35,35,35]` |

So Gate C on GDF compared two DIFFERENT XC quadratures. On FFTDF the DF mesh
happens to equal `cell.mesh`, which is exactly why the FFTDF arm of the
"identical" comparison passed and why only GDF failed. The same line exists in
`kuks.rs:80`, `kroks.rs:63` and `kgks.rs:63`, so every non-FFTDF
`K{R,U,RO,G}KS::from_df` integrates XC on the DF's auxiliary mesh.

17-08's leading hypothesis ("GDF's `_cderi` is not symmetry-invariant, so the
full-BZ GDF solution is symmetry-broken") is **refuted**: step 1 is exactly 0.

## Task 1 — non-vacuity (asserted in `fixture()` before any number is read)

| check | value |
|---|---|
| `nkpts` | **8** |
| `nkpts_ibz` | **3** (`assert!(nibz < nk)`) |
| `ibz2bz` / `weights_ibz` | `[0, 6, 7]` / `[0.125, 0.375, 0.5]` |
| band set = `kpts_ibz`, strict subset | `3 < 8` asserted; each `kpts_ibz[i]` asserted **bitwise** equal to `kpts[ibz2bz[i]]` |
| one density for every arm | converged full-BZ FFTDF `KRKS` (`e_tot -7.772967813414`), symmetrised `D_sym = unfold(D[ibz2bz])`; `D_sym[ibz2bz] == D_ibz` **0e0** and `unfold(D_sym[ibz2bz]) == D_sym` **0e0** (both asserted) |
| symmetry breaking of the raw converged density | `max|D − D_sym|` = 5.434e-10 (default `cell.precision`; the Gate-B floor, not used downstream) |

The density is the same bits in both arms, so no row below measures SCF drift.
The XC rows are first-order in the grid, so a non-GDF-converged density changes
nothing about which step breaks.

## Task 2 / Task 3 — every step, GDF and the FFTDF control, one density

Fixture `si()` default precision, `[2,2,2]`, `lda,vwn`, `exxdiv = ewald`, both
arms built exactly as `krks_ksymm.rs::krks_ibz_energy_matches_full_bz_on_gdf`
builds them. `max|Δ|` is over all IBZ points, `re` and `im`.

| step | quantity compared | GDF `max|Δ|` | FFTDF `max|Δ|` |
|---|---|---:|---:|
| 1a | `_cderi`, the two arms' independent full-BZ builds, all 64 `(ki,kj)` blocks | **0e0** | n/a |
| 1b | `_cderi` `(k,k)` blocks, full-BZ fit vs a fit built over the 3 IBZ points only (ranks/`nao_pair` equal: `(90,36)`) | **0e0** | n/a |
| 2 | `get_j`, full-BZ direct vs the ksymm `kpts_band = kpts_ibz` path (`veff::get_jk`) | **0e0** | **0e0** |
| 2e | `ecoul`, `1/nk` trace vs `weights_ibz` trace | 9.090e-12 | 6.739e-14 |
| 3 | `get_k` (unused by LDA), direct vs `kpts_band` path | **0e0** | **0e0** |
| 4 | `kpts_band` route vs direct on ONE DF object, `vj` / `vk` | **0e0 / 0e0** | **0e0 / 0e0** |
| 5 | `hcore` at the full set vs at the IBZ set | **0e0** | **0e0** |
| **6** | **`vxc`, each arm on its own XC grid** | **1.810e-05 — FIRST > 1e-9** | 3.331e-16 |
| 6e | `exc`, each arm on its own XC grid (`-2.378189945851` vs `-2.378188513928`) | **1.432e-06** | 3.997e-15 |
| 7 | `E_elec[D]`, production `get_hcore`/`get_veff`/`energy_elec` of both drivers as the gate builds them | **1.432e-06** | 1.008e-13 |
| 8 | `vxc`, full arm's grid replaced by `cell.mesh` | 3.331e-16 | 3.331e-16 |
| 8e | `exc`, matched grid | 3.997e-15 | 3.997e-15 |
| 8E | `E_elec[D]`, matched grid | **2.719e-10** | 1.008e-13 |

Reading it:

* Steps 1-5 — the GDF fit, J, K, the `kpts_band` rebuild route and hcore — are
  **bit-identical** between the IBZ and full-BZ arms. Nothing GDF-specific
  differs. Step 4 re-confirms 17-08's `gdf_band_route_matches_the_direct_route`
  (0e0) at the symmetrised density.
* Step 6e reproduces the gate's `1.432444577176e-06` to four digits
  (`1.43192e-06` at this density; step 7, the full energy functional,
  `1.43165e-06`). The whole discrepancy is `exc`.
* Step 8 removes it by changing ONE thing — the full arm's XC grid — and
  touching no GDF code: `vxc` falls from 1.8e-05 to 3.3e-16 and `E_elec[D]`
  from 1.432e-06 to **2.719e-10**, inside the Gate-C GDF `KRHF` floor
  (`2.486e-10`, `17-VERIFICATION.md:124`) and below the plan's 1e-9 threshold.
  (By subtraction the residual 2.7e-10 is in the `e1` trace — `exc` is at
  4e-15 and `ecoul` at 9.1e-12, and every operator is bit-identical — i.e.
  in `D_sym` at the non-IBZ points against `hcore` at DEFAULT
  `cell.precision`. Not separately printed.)
* The FFTDF control resolves 1e-13 at every step on the same harness, so the
  harness is not the source of a 1e-6 (Task 3's requirement). Its `E_elec[D]`
  1.008e-13 is the default-precision analogue of the recorded 3.109e-14.

## Confirmation at the gate's own level (three GDF SCFs)

`gdf_gate_c_with_and_without_matched_xc_grid` — the gate re-run as written, and
again with ONLY `full.grids = PeriodicGrids::uniform(&cell, Some(cell.mesh))`.

exit=0, 2543.6 s (concurrent builds on the host, load ~25):

| arm | XC grid | `e_tot` | `|dE|` vs `e_ibz` |
|---|---|---|---|
| IBZ `KsymAdaptedKrks` (GDF) | `[35,35,35]` | `-7.774588786147` | — |
| full-BZ `Krks::from_df(Gdf)`, **as written** | `[13,13,13]` | `-7.774590218592` | **1.4324445718e-06** |
| full-BZ, **grid matched** to `cell.mesh` | `[35,35,35]` | `-7.774588785875` | **2.7204905e-10** |

The as-written run reproduces 17-08's recorded `e_full`, `e_ibz` and
`1.432444577176e-06` to all 12 printed digits, so the defect is present in
today's tree unchanged. With the one grid line matched the converged Gate C is
**2.720e-10** — under the gate's 1e-8 by 37×, under the plan's 1e-9, and
within 1.1× of the GDF `KRHF` Gate C (`2.486e-10`), which has no XC grid at
all. It also equals step 8E's fixed-density `2.719e-10`, as a variational
energy should.

## Classification — (a) defect

* **Not (b).** A fitting difference would show at step 1 or 2; both are 0e0,
  including the fit built over the IBZ points alone (1b). The GDF fit is
  identical on either k-set for the blocks J uses.
* **(a), located:** `crates/pyscf-pbc-dft/src/krks.rs:93` (and
  `kuks.rs:80`, `kroks.rs:63`, `kgks.rs:63`) take the XC grid mesh from
  `with_df.mesh()` instead of `cell.mesh`. Expected `[35,35,35]` (upstream
  `UniformGrids(cell)`); actual `[13,13,13]` for `Gdf`. Consequence beyond
  Gate C: any full-BZ KS SCF over GDF/RSDF/MDF/AFTDF built via `from_df` has an
  XC quadrature error set by the DF's auxiliary mesh — on this fixture
  **1.43e-06 Ha** in `exc`, and `nelec` off by 6.1e-10 vs 2.6e-11.
* **The gate is right and stays as written** (1e-8, `GDF_E_TOL`,
  `krks_ksymm.rs:735`); no restatement of the gate. A row pointing here is
  APPENDED to `measurements/README.md` §1 (row 11 itself is not rewritten).

## Hand-off to the fix (NOT done here)

1. The obvious edit — `Some(with_df.cell().mesh)` — **changes behaviour for
   existing FFTDF callers that pin a DF mesh different from `cell.mesh`**:
   `tests/gate.rs:46`, `tests/modules.rs:34` and
   `pyscf-bench/src/bin/krks_repro.rs:83` use `Fftdf::with_mesh(cell, kpts,
   mesh)` and currently get an XC grid on that pinned mesh. Upstream's
   equivalent pins `cell.mesh`, so the faithful fix either pins `cell.mesh` in
   those callers or keeps the DF mesh only for `FFTDF`. Re-run those gates.
2. After the fix, `krks_ibz_energy_matches_full_bz_on_gdf` should be re-run
   unmodified and its doc comment's "STATUS: RUN, AND IT FAILS" block and
   `17-VERIFICATION.md:126` / §10.8 updated with the new number.
3. Grep for other `with_df.mesh()` grid constructions before closing
   (`kuks`/`kroks`/`kgks` listed above).

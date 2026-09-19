# 20-13 SUMMARY — `pbc.dft`: KRKS/KUKS/KROKS/KGKS + 4 ksymm + DFT+U + numint backends + grids

**Shipped:** 2026-09-14. All six tasks are done. Task 1's feature posture is superseded by D-20-B. Two `pyscf-pbc-*` crate defects were found and are **refused in the binding, not fixed**: KUKSpU's Hubbard formula, and ksymm GGA on an s-only basis. See D3 and D4. Every plan verification command is green except where a deviation says otherwise.

## Task 1 — libxc posture (superseded by D-20-B)

- `crates/pyscf-py/Cargo.toml` is **untouched**. `pyscf-pbc-dft` was already a default-features dependency (20-09 Step 0a). No `libxc` feature was added.
- `cargo tree -p pyscf-py -e normal | grep -c libxc` gives **547**, unchanged from 20-09.
- `cargo tree -e features` shows `pyscf-pbc-dft feature "default"` → `"libxc"` and `pyscf-dft feature "libxc"`. No xcfun default was introduced; `XcBackend::default()` is libxc (`pyscf-dft/src/xc_backend.rs:211`).
- Empirical check: KRKS Si PBE through the binding lands **6.45e-12** from upstream running its libxc default (asserted `xclib == pyscf.dft.libxc`). xcfun would sit 4.7e-7 away (12-VERIFICATION §1e), so the escalation trigger did not fire.
- Warm pyscf-py-only rebuilds: cargo `Finished` 27.31 s, 23.99 s, 23.53 s, 30.26 s (`target/py-20-13-build{1..4}.log`, maturin exit 0 ×4).

## Task 2 — tests first

`python/pyscf/tests/test_pbc_dft.py` has **26 cases**.
- **RED:** collection failed on the pre-20-13 `.so` with `AttributeError: module 'pyscf._native.pbc.dft' has no attribute 'KRKS'`.
- **Upstream oracle:** one vendored-2.12.1 subprocess, with the version and the libxc backend asserted. It runs KRKS Si PBE (the `gate.rs` oracle script) and KRKSpU He.

## Tasks 3–6 — the binding (`crates/pyscf-py/src/pbc/dft.rs`, 20-12's pattern)

| Python | Rust (named by module; the crate has no root re-exports) | base |
|---|---|---|
| `KohnShamDFT` | — (state, the 11 hooks, `kernel`, results) | `#[pyclass(subclass, dict)]` |
| `KRKS(cell, kpts=None, xc='LDA,VWN', exxdiv='ewald')` | `krks::Krks`; with a `KPoints` → `krks_ksymm::KsymAdaptedKrks` | `KohnShamDFT` |
| `KsymAdaptedKRKS(cell, kpts, xc, exxdiv, use_ao_symmetry)` | `KsymAdaptedKrks` | `KRKS` |
| `KUKS(...)` | `kuks::Kuks`; with a `KPoints` → `KsymAdaptedKuks` | `KohnShamDFT` |
| `KsymAdaptedKUKS(...)` | `KsymAdaptedKuks` (no `from_df`: built with `new`, then the pub `with_df`/`grids` are swapped in) | `KUKS` |
| `KROKS(...)` / `KGKS(...)` | `kroks::Kroks` / `kgks::Kgks`; a `KPoints` RAISES | `KohnShamDFT` |
| `KRKSpU(cell, kpts, xc, exxdiv, U_idx, U_val, C_ao_lo, minao_ref)` | `Krks` + `kspu::add_vhubbard` (`PuDriver`); with a `KPoints` → `KsymAdaptedKrkspu` | `KRKS` |
| `KUKSpU(...)` | **refuses at use** (D3); with a `KPoints` → `KsymAdaptedKukspu`, also refused | `KUKS` |
| `KsymAdaptedKRKSpU` / `KsymAdaptedKUKSpU` | `krks_ksymm::KsymAdaptedKrkspu` / `KsymAdaptedKukspu` | `KsymAdaptedKRKS` / `KsymAdaptedKUKS` |
| `UniformGrids(cell)` (`mesh`, `size`, `coords`, `weights`, `build`) | `gen_grid::PeriodicGrids::uniform` | — |
| `BeckeGrids(cell)` (`level`, `build`, `size`, `coords`, `weights`) | `PeriodicGrids::becke` (cached until `level` or `cell` changes) | — |
| `KNumInt` / `MultiGridNumInt` / `MultiGridNumInt2` (`mesh`) | `numint::KsNumInt::grid` / `::multigrid` / `::multigrid2` | — |
| functions `RKS`/`UKS`/`ROKS`/`GKS(cell, kpt, xc, exxdiv)` | the K-driver at one k-point (`gamma.rs`) | — |

### How the base class works

- **Per call.** An `enum Driver` (10 variants) is built from `extract_df(with_df)`, the stored configuration, `mf.grids` and `mf._numint`. `impl KOverrideHooks for Driver` is pure delegation, and `KPyOverrideBridge::new(.., &py.get_type::<PyKohnShamDft>(), ..)` probes against the base.
- **The default grid path is not reconstructed.** A `UniformGrids` with no explicit `mesh` means "keep `from_df`'s own `PeriodicGrids::uniform(cell, None)`". That is the 20-04-FIX grid on `cell.mesh`; `mf.grids.mesh = X` pins it.
- **Energy-tag guard (`TagGuard`) — found by the override test.**
  - Problem: a Python override of `get_veff` bypasses the Rust `get_veff`, so the KS `ecoul`/`exc` tags go stale. `Krks::energy_elec` recomputes them once and then reuses the cycle-1 values. Measured on He: **|dE| = 6.2e-3 Ha** against the unbridged kernel.
  - Fix: the guard sits between the bridge and the driver. When an `energy_elec` arrives without a preceding `get_veff` through the guard, it refreshes the tags on the same density. This is upstream's "untagged `vhf` → recompute" rule (`krks.py:118-119`).
  - With no override it makes no extra call. After the fix, the override-then-kernel result is **bitwise** equal to the unbridged kernel.
- **Attributes, get/set:**
  - `xc`, `nlc` (only `''`), `grids`, `_numint`, `exxdiv`
  - the SCF controls of 20-12: `conv_tol`, `conv_tol_grad`, `max_cycle`, `diis*`, `damp`, `level_shift`, `init_guess`, `chkfile` (written by `kernel`), `verbose`
  - per-driver attributes that raise `AttributeError` elsewhere: `nelec` (U/RO), `init_guess_breaksym` (U), `use_ao_symmetry` (ksymm), `collinear` (KGKS), `U_idx`/`U_val`/`minao_ref`/`alpha`/`C_ao_lo` (+U)
- **Results:** `e_tot`, `e_elec`, `e_coul`, `e_nuc`, `converged`, `cycles`, `fermi`, `mu`, `e_free`, `e_zero`, `mo_energy`, `mo_occ`, `mo_coeff` (20-11 codec), and `e_u`.
- **Hooks** as in 20-12. On top of those:
  - `get_veff(kpts_band=)`;
  - `get_rho(dm)` (spin-summed for U/RO; grid numint only);
  - `get_bands` (KRKS/KUKS);
  - `smearing_` (KRKS/KUKS/+U);
  - `multigrid_numint(mesh=None)` (in place, returns self);
  - `density_fit()` (GDF with `_j_only` for a pure functional, plus `BeckeGrids`, as `_patch_df_beckegrids` does);
  - `nr_fxc(dms, dm0, hermi)` (KGKS);
  - private `_veff_components(dm)` → `{vxc, ecoul, exc, nelec, E_U}` (upstream's `tag_array` attributes);
  - `_kernel_without_bridge()`, `_overridden_hooks`, `_is_ksymm`.
- **DFT+U sites.** `U_idx` labels `'El nl'` map to `USite::Shell{element, l, contraction: n-l-1}`. The following raise `NotImplementedError`:
  - integer AO lists (upstream maps large-basis indices through AO labels, `rkspu.py:156-158`; the Rust `USite::Indices` indexes the MINAO basis);
  - atom-index prefixes;
  - explicit `C_ao_lo` arrays;
  - a per-site `alpha` list.
- **Python overlay** (`python/pyscf/pbc/dft/__init__.py`):
  - re-exports the native objects (identity) and ports `RKS`/`UKS`/`GKS`/`ROKS`/`KS`/`KKS` (`__init__.py:76-119`);
  - does **not** import upstream's dft `__init__`;
  - `pyscf.pbc.scf.KRKS(...)` (20-12's forward) now returns a native `KRKS`.

## Verification (2026-09-14, final `.so` from build 4)

| command | result |
|---|---|
| `.venv/bin/pytest python/pyscf/tests/test_pbc_dft.py -q -p no:cacheprovider` | **26 passed in 243.64 s**, exit 0 (`target/py-20-13-test2.log`) |
| KRKS Si 2×2×2 PBE, gth-szv, mesh 31 (`gate.rs` fixture), binding vs Rust `kernel()` **bitwise** | Checked two ways. (1) `kernel()` (bridge) vs `_kernel_without_bridge()` (concrete `Krks::kernel`) on a fresh object: `e_tot`, `cycles`, `mo_energy` and `mo_coeff` all bit-identical. (2) vs the value `gate.rs` printed for the identical fixture (`rust -7.785668903719571`, `target/p20-02-logs/dft__gate__krks_si_222_pbe_matches_upstream.log`): the binding prints the same 15 decimals, and the float residual is 8.88e-16 (1 ulp, the decimal rounding) |
| KRKS Si PBE vs upstream (live, 5.5 s upstream) | **6.449064e-12** and 6.450840e-12 on two runs; upstream's last digit varies between runs. Floor row 3 is 6.45e-12; gate 1e-11. `e_nuc` equal to 1e-12 |
| KUKS / KROKS / KGKS on closed-shell He PBE vs KRKS | `|dE| = 0.0` each; each bridged run bitwise equal to its unbridged run |
| Python `get_veff` override | `_overridden_hooks == ['get_veff']`, and bitwise equal to the unbridged KRKS (after `TagGuard`) |
| ksymm Gate C, FFTDF, diamond-structure Si PBE, cell mesh 35³, 3 of 8 k | KRKS **1.971756e-13**, KUKS **1.838529e-13** (asserted `< 1e-10`). Explicit `KsymAdaptedKRKS`/`KsymAdaptedKUKS` and `_kernel_without_bridge` are bitwise equal to `KRKS(cell, kpts=KPoints)` |
| ksymm Gate C, GDF (20-04-FIX: MET 2.1997e-10 on si) | He sto-3g LDA `|dE| = 0.0` (asserted `< 1e-8`, the Gate-C bound). Both arms' `grids.mesh == cell.mesh`. D6 explains why this fixture is weak |
| DFT+U `E_U`, IBZ vs full BZ (17-08 gate) | **6.938894e-18** (17-VERIFICATION 6.939e-18); `E_U = 0.026457839749303`, matching 17-08's print |
| KRKSpU He SCF | bridged bitwise equal to unbridged; `E_U = -2.04e-17` on the filled shell (17-08's number); `|E - E_KRKS| = 0` |
| KRKSpU He vs upstream | `e_tot` **5.995e-14** (asserted `< 1e-12`). `E_U` at 0.35 occupancy: **2.197e-10** (0.026457839749303 vs 0.026457839969033). First measurement; asserted `< 1e-9`. See D7 |
| numint backends, diamond gth-szv gamma, mesh 25, random density | `ecoul` \|v1−grid\| **1.12e-14**; \|v2−grid\| **8.09e-10** (asserted `< 2e-7`, the upper end of v2's own floor, never v1's). `exc` \|v1−grid\| 3.55e-15; \|v2−grid\| **7.91e-7** (reproduces 17-VERIFICATION's Δexc ≤ 7.9e-7; not asserted against a floor) |
| multigrid v2 SCF, Si gamma, mesh 11 | \|E_v2 − E_grid\| **2.226e-7** (`multigrid_scf.rs` gate 2e-3) |
| refusals raise | Covered cases: KGKS + `pbe0` (`NotYetImplemented`, "hybrid"); `collinear='mcol'` (`NotYetImplemented`); `'ncol'` + PBE (`NotYetImplemented`); `nr_fxc` with a non-collinear treatment (`NotYetImplemented`). Also: multigrid v1 at 8 k ("gamma"); v2 + hybrid; KGKS + multigrid; multigrid mesh ≠ `cell.mesh`; KROKS/KGKS + `KPoints`; `nlc`; KROKS `get_bands`; unported `U_idx`/`C_ao_lo`; KUKSpU (D3); ksymm GGA on an s-only basis (D4) |
| `pytest test_pbc_identity_gate.py` | **18 passed**: the dft ×4 now pass, and mp/cc pass through the concurrent 20-15 |
| `check-catch-unwind` / `check-forbid-lazy-static` / `check-dependency-wall` / `check-orphan-modules` | exit 0 (849 files) / 0 / 0 (both PASS) / 0 (449 files) |
| `cargo clippy -p pyscf-py --release` | no warning in `pbc/dft.rs` |
| full `.venv/bin/pytest python/pyscf/tests -q -p no:cacheprovider` | **11 failed, 321 passed, 4 skipped, 1 deselected, 3 xfailed, 9 errors in 559.02 s** (exit 1, `target/py-20-13-fullsuite.log`). `test_pbc_dft` 26/26 and `test_pbc_identity_gate` 18/18 pass. The one PBC failure is `test_pbc_post_scf.py::test_ksymm_kmp2_fftdf_vs_full_bz`, which belongs to the concurrent 20-15 agent's in-flight file; it does not touch `pbc/dft.rs`. The other 10 failures and 9 errors are the known molecular ones from `measurements/python-suite-triage.md`: `test_panic_to_exception` ×2, `test_scf_cross_dispatch` ×3, `test_scf_ghf`, `test_scf_rhf_ccpvdz` ×3 and `test_scf_rhf_h2o`, plus 9 R1 overlay-import errors. Against 20-12 (16 failed / 263 passed): the 6 identity-gate failures are gone, and +58 passes come from `test_pbc_dft` and 20-15's files. |

## Deviations

- **D1 — `KRKS` is not a subclass of `pyscf.pbc.scf.KRHF`.** The scf native base (`KSCF`) belongs to 20-12, and its hook bodies know only the HF drivers. `isinstance(KRKS(...), KRHF)` is therefore False, whereas it is True upstream. As in 20-12 D1, the `KPoints` dispatch lives in the native constructors, so `type(KRKS(cell, kpts=KPoints))` is `KRKS`, not `KsymAdaptedKRKS`.
- **D2 — the bridge wraps the driver in `TagGuard`** (see above). The guard is binding-side. The underlying issue is that the Rust KS drivers keep energy tags in interior `Cell`s. That contract cannot survive a hook being served by Python.
- **D3 — CRATE DEFECT: `kspu::add_vhubbard(_weighted)` is wrong for KUKSpU.**
  - What the Rust code does: it applies the restricted `E_U = w(U/2)(Tr P − Tr P²/2)`, `V = (1−P)U/2` to each spin channel (`kspu.rs:536-545`). Its doc argues this is correct "per CHANNEL".
  - What upstream does: `kukspu.py:98-99` uses `Tr P − Tr P²` and `(1 − 2P)U/2` per spin.
  - Measured, vendored 2.12.1, He sto-3g 2×2×2, U = 5 eV on `He 1s`:

    | density | upstream `E_U` | Rust-formula `E_U` |
    |---|---|---|
    | converged | **5.6e-14** | **9.187e-02** |
    | 0.35 per spin | **4.1514e-02** | **5.2916e-02** (2 × the restricted value) |

  - The 17-08 `E_U` gate only exercises the restricted form, and `KsymAdaptedKukspu` plus `pyscf-pbc-grad/src/stress/kuks.rs:656` share the routine.
  - The plan rule is not to edit pbc crate src. `KUKSpU` and `KsymAdaptedKUKSpU` therefore construct, dispatch and report attributes, but **every compute path raises `NotImplementedError`** naming `kukspu.py:98-99`. **Needs a `pyscf-pbc-dft` fix, and then a Phase-18 stress re-check.**
- **D4 — CRATE DEFECT: an interpreter abort.** `KPoints::symmetrize_density_vec` (`pyscf-pbc-symm/src/kpts.rs:2121`) indexes `dmats()[iop][1]`. On an s-only basis the `Symmetry` Wigner-D sets stop at `l = 0`, so ksymm + GGA (the default S-03 symmetrized quadrature) panics with index out of bounds. The release profile is `panic = "abort"` (`Cargo.toml:138`), so this **killed the pytest process** (measured with He sto-3g PBE). The binding now raises `NotImplementedError` before the call when all of these hold: ksymm, grid numint, GGA, some `dmats` entry shorter than 2, and `PYSCF_PBC_KSYMM_RHO` not set to `unfold`. The ksymm GGA tests use diamond-structure Si (gth-szv has p functions). **Needs a pbc-symm fix: build `l = 1` whenever a GGA vector density is symmetrized.**
- **D5 — a pinned coarse mesh is not a Gate-C fixture.** The first ksymm tests pinned `with_df.mesh = grids.mesh = [15]*3` and measured IBZ − full-BZ gaps of **1.37e-6** (He cc-pvdz, LDA and PBE, with and without `PYSCF_PBC_KSYMM_RHO=unfold`). The gap shrinks with the mesh:

  | He sto-3g LDA mesh | IBZ − full-BZ gap |
  |---|---|
  | 15³ | 8.31e-7 |
  | 16³ | 5.21e-7 |
  | 21³ | 2.94e-9 |
  | default 43³ | **8.39e-14** |

  This is quadrature aliasing, not a symmetry defect: rotations do not commute with an under-resolved grid. The tests use each cell's own symmetrized mesh. No bound was loosened.
- **D6 — GDF ksymm through the binding is expensive, so the suite fixture is weak.**

  | run | `|dE|` | notes |
  |---|---|---|
  | He sto-3g (1 AO, 2 cycles), in suite | exactly 0.0 | fixture is weak |
  | He 6-31g (2 AO), ad hoc | **9.33e-15** | 125 s, 99³ XC grid; not in suite |
  | diamond-structure Si GDF LDA | outcome not observed | killed after >600 s (D-20-D rule) |

  The 20-04 property the defect broke (the XC grid follows `cell.mesh`, not the DF mesh) is asserted directly. The Rust-side MET number (2.1997e-10) belongs to 20-04-FIX.
- **D7 — KRKSpU `E_U` sits 2.2e-10 from upstream at a fractional density, while `e_tot` agrees to 6e-14.** The local orbitals are built by `zsolve_linear` + `zeigh_gen`-based Löwdin here and by `cho_solve` + `vec_lowdin` upstream (`kspu.rs:274-414`). On a filled shell the energy is insensitive to that. The number is recorded as a first measurement and bounded at 1e-9, not at a floor.
- **D8 — upstream parity gaps:**
  - `multigrid_numint`, `smearing_` and `density_fit` mutate in place and return self (upstream returns a new object);
  - multigrid refuses a `mesh` other than `cell.mesh` (the Rust engines integrate on `cell.mesh`);
  - `get_rho` works with grid numint only (`KsNumInt::get_rho`);
  - `get_bands` exists for KRKS/KUKS only;
  - `init_guess='chkfile'` is not bound; `chkfile` is written, not read;
  - `energy_elec(vhf_kpts=)` ignores `vhf` and recomputes the tags on `dm`;
  - `BeckeGrids` binds `level` only;
  - `nlc`/VV10 is refused;
  - gamma shims return per-k lists of length 1;
  - `kernel(**kwargs)` raises `TypeError`.
- **D9 — upstream submodules via `pkgutil`, recorded for 20-17.** With `PYTHONPATH=python` and the overlay active, **all 19** `pyscf.pbc.dft.<sub>` fall through to the vendored files, and **all 19 fail import** before reaching any 20-12 stub:
  - `ATM_SLOTS` missing from `pyscf.gto`: `gks`, `kgks`, `krks`, `krkspu`, `krkspu_ksymm`, `kroks`, `kuks`, `kukspu`, `kukspu_ksymm`, `rks`, `uks`;
  - `mole`: `cdft`, `krks_ksymm`, `kuks_ksymm`;
  - `radi`: `gen_grid`;
  - `numint` from `pyscf.dft`: `numint`, `numint2c`;
  - `rohf` from `pyscf.scf`: `roks`;
  - `ATOM_OF`: `multigrid`.

  So `kuks_ksymm`'s dependency on the `kuhf_ksymm` stub (20-12 D5) is not observable today. The overlay serves the native classes at the top level, and upstream's `pbc/dft/__init__.py` is never executed. Also observed: running from `cwd=python/` **without** `PYTHONPATH` imports `.venv` site-packages pyscf **2.14.0** and bypasses the overlay entirely. The test `conftest.py` handles this.
- **D10 — edits outside the three owned files:** `crates/pyscf-py/src/pbc/mod.rs` got 3 additive lines (`pub mod dft;`, a comment, and `"dft" => dft::register(&m)?`). The concurrent 20-15 agent's lines (`ao2mo`/`cc`/`ci`/`mp`) were left alone. The module `__doc__` is overwritten inside `dft::register`, so 20-08's doc match arm was not touched.
- **D11 — CubeCL manual (AGENTS.md §3) not consulted:** no compute kernel was written. This plan is PyO3 glue plus delegation over existing drivers.

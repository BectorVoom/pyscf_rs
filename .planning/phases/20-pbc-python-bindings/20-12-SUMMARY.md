# 20-12 SUMMARY — `pbc.scf`: KRHF/KUHF/KROHF/KGHF + ksymm + smearing + chkfile + `get_bands`

**Shipped:** 2026-09-14. All six tasks done. Every plan verification command is green; deviations below.

## Task 1 — the tests

`python/pyscf/tests/test_pbc_scf.py` (21 cases). **Not strictly RED-first:** the file was written while the first `.so` with the drivers was compiling, and was never run against the pre-20-12 `.so`. Its first run failed 2 cases for real reasons: `get_grad` on the 1-AO fixture is empty (test assumption fixed), and `KRHF(cell, kpts=KPoints)` refused with `use_ao_symmetry = true needs cell.symm_orb` (binding fixed, D2).
Fixture He-fcc all-electron Bohr; `sto-3g` 2×2×2 with `with_df.mesh = [15]*3` is **exactly** `krhf_bands_oracle.rs`'s fixture and settings (`conv_tol 1e-12`, `conv_tol_grad 1e-8`, `max_cycle 60`); `cc-pvdz` where one AO is vacuous (smearing needs virtuals, projection needs two AO counts). One upstream subprocess (vendored 2.12.1, version asserted) supplies every upstream number.

`test_pbc_override_dispatch.py`: `DRIVERS = [_KRHFBridgeSelftest, KRHF]`; three adapters (`new_driver`, `run`, `run_reference`) map the harness's `cls(with_df)` / `kernel(conv_tol=, use_bridge=)` shape onto upstream's `KRHF(cell, kpts)` + attributes; assertions unchanged. The `needs 20-12 KRHF` skip is gone.

## Tasks 2–3 — the drivers (`crates/pyscf-py/src/pbc/scf.rs`)

| Python | Rust | base |
|---|---|---|
| `KSCF` | — (state, the 11 hooks, `kernel`, results) | `#[pyclass(subclass, dict)]` |
| `KRHF(cell, kpts=None, exxdiv='ewald')` | `Krhf`; a `KPoints` → `KsymAdaptedKrhf` | `KSCF` |
| `KsymAdaptedKRHF(cell, kpts)` | `KsymAdaptedKrhf` | `KRHF` |
| `KROHF(...)` | `Krohf` | `KRHF` |
| `KUHF(...)` / `KGHF(...)` | `Kuhf` / `Kghf`; a `KPoints` RAISES | `KSCF` |

- **Pattern (20-13 should copy it).** Everything lives on one native base; the public classes are `extends` subclasses with a `#[new]` only (PyO3 without `multiple-pymethods` allows one `#[pymethods]` per class). `KPyOverrideBridge::new(py, slf, py_cell, &py.get_type::<PyKscf>(), &driver)` — the probe base is the class that DEFINES the hooks, so `KRHF`/`KUHF`/… instances resolve to it and report no override.
- **Per call:** `enum Driver { Rhf, KsymRhf, Uhf, Rohf, Ghf }` is built from `extract_df(with_df)` + stored config (`exxdiv`, `smearing`, `nelec`, `init_guess_breaksym`, `use_ao_symmetry`) and `impl KOverrideHooks for Driver` delegates every trait method (macro `each_driver!`), so the bridged kernel is op-for-op the concrete kernel. Only `KScfResult` (+ nao/nfock/kpts) is stored.
- **Ownership:** `with_df` is the Python object (`mf.with_df is mydf`, settable, validated by `extract_df`); default is upstream's `FFTDF(cell, kpts)`. `mf.kpts` reads `with_df.kpts` (the `KPoints` for ksymm); setting it re-targets `with_df.kpts`.
- **Config attributes** (get/set): `conv_tol` (default `KScfConfig::for_cell`), `conv_tol_grad` (None = sqrt), `max_cycle`, `diis` (bool), `diis_space`, `diis_start_cycle`, `damp`, `level_shift`, `init_guess` (`minao`/`atom`/`1e`/`chkfile`), `chkfile`, `verbose`, `exxdiv`; `nelec` (KUHF/KROHF), `init_guess_breaksym` (KUHF), `use_ao_symmetry` (ksymm).
- **Results:** `e_tot` (0.0 before), `e_elec`, `e_coul`, `e_nuc`, `converged`, `cycles`, `fermi`, `mo_energy`/`mo_occ` (per-k float lists, nested per spin for KUHF), `mo_coeff` (per-k complex `(nao,nmo)` via the 20-07/20-11 codec), smearing `mu`, `e_free`, `e_zero`, `entropy`.
- **Hooks** with upstream signatures and upstream defaults (`None` → stored result / `get_hcore()` / `get_veff(dm)`): `get_ovlp(cell, kpts)`, `get_hcore(cell, kpts)`, `get_init_guess(cell, key, s1e)`, `get_veff(cell, dm_kpts, …)`, `get_fock(h1e, s1e, vhf, dm, cycle=-1, …)`, `eig`, `get_occ`, `make_rdm1`, `energy_elec`, `energy_nuc`, `energy_tot`, `get_grad(mo_coeff, mo_occ, fock=None)`.
- `kernel(dm0=None)` → `e_tot` (bridge; a second call restarts from the stored wavefunction, `hf.SCF.scf`), `scf`, `run(**attrs)` → self, private `_kernel_without_bridge()` (concrete `Krhf::kernel`, the bitwise reference) and `_overridden_hooks`. GIL held throughout (drivers not `Sync`).

## Task 4 — `get_bands`, smearing, chkfile, gamma shims

- `get_bands(kpts_band, cell=None, dm_kpts=None)` — KRHF/KUHF (`Krhf::get_bands`, `Kuhf::get_bands`); `(3,)` → single arrays. KROHF/KGHF/ksymm raise `NotImplementedError` (no Rust `get_bands`).
- `mf.smearing_(sigma, method='fermi'|'gaussian', mu0)` → self (KRHF/KUHF; others raise), alias `smearing`, module `smearing_(mf, …)`; `sigma`, `smearing_method`.
- `mf.chkfile = path` → `kernel()` writes `dump_kscf_to_file` (+ `/mol` = `cell.dumps()`); `dump_chk(path)`; module `load_scf(path)` → `(Cell | None, {e_tot,kpts,mo_energy,mo_occ,mo_coeff})`; `init_guess_by_chkfile(chk, project)` / `from_chk` / `init_guess='chkfile'` (stored k-points must match; projects across a basis change).
- Gamma shims: native `RHF/UHF/ROHF/GHF(cell, kpt=None, exxdiv)` = the K-driver at one k-point (`gamma.rs`).

## Task 5 — `project_mo_nr2nr` IMPLEMENTED (`crates/pyscf-pbc-scf/src/addons.rs`)

`project_mo_nr2nr(cell1, mo1: &[CTensor], cell2, kpts) -> Result<Vec<CTensor>>`: `S22 = get_ovlp(cell2)`, `S21 = intor_cross("int1e_ovlp", cell2, cell1, hermi 0)`, per-column `zsolve_linear` (upstream: Cholesky `solve(assume_a='pos')`). The `phase: 20` refusal and module note are gone. Consumer: `init_guess_by_chkfile` across a basis change. Rust test `crates/pyscf-pbc-scf/tests/project_mo_nr2nr.rs` (3 cases: same-cell identity `2.78e-16`; normal equations `S22 C2 = S21 C1` both directions `≤1.1e-16` on a non-TRIM 3×1×1 mesh with `max|Im S22| = 6.1e-2`; norm never grows; count/shape mismatch errors). Bound as `pyscf._native.pbc.scf.project_mo_nr2nr`.

## Task 6 — the `KPoints` dispatch, and the missing ksymm variants

- `KUHF`/`KGHF` with a `KPoints` (positional or keyword, native or overlay, and overlay `UHF(cell, kpts=KPoints)`) raise `NotImplementedError: … not implemented in pyscf-rs; upstream has pbc/scf/kuhf_ksymm.py` (resp. `kghf_ksymm.py`). `KROHF` + `KPoints` raises too.
- `python/pyscf/pbc/scf/__init__.py`: re-exports the native classes (identity) and ports the function dispatch `RHF` (`kpts=` → KRHF; spin 0 → gamma RHF; else ROHF), `UHF`, `GHF`, `ROHF`, `HF`, `KHF`, and the `KS`/`KKS`/`RKS`/… forwards to `pyscf.pbc.dft`. `pyscf.pbc.scf.kuhf_ksymm` / `.kghf_ksymm` are `sys.modules` stubs (`__file__` absent) whose public attributes raise `NotImplementedError`; dunder probes raise `AttributeError` so `inspect`/pytest stay safe. Other upstream submodules fall through lazily (20-17).
- `KRHF(cell, kpts=KPoints)` runs `KsymAdaptedKrhf`; its default FFTDF is built over a **copy** of the cell with `basis::build_symmetry` applied (`use_ao_symmetry=True` needs `cell.symm_orb`).

## Verification (2026-09-14)

| command | result |
|---|---|
| `maturin develop --release --skip-install` (EXECUTION-NOTES §2 env) | maturin exit=0 ×3: `Finished` 3m00s (pbc-scf change cascades), 20.87 s, 17.91 s (final `.so`, logs `target/py-20-12-build{1,2,3}.log`) |
| `.venv/bin/pytest python/pyscf/tests/test_pbc_scf.py -q -p no:cacheprovider` | **21 passed in 81.32 s**, exit 0 |
| KRHF He-fcc 2×2×2 through the binding == Rust `kernel()` **bitwise** | (1) `kernel()` (bridge) vs `_kernel_without_bridge()` (concrete `Krhf::kernel`) on a fresh object: `e_tot`, `cycles`, `mo_energy`, `mo_coeff` all bit-identical; (2) `f"{e:.15f}" == "-2.807388116559963"`, the value `krhf_bands_oracle.rs` prints (`target/gate-20-12-bands.log`) — equal at its printed precision (`e = -2.8073881165599626`) |
| KRHF vs upstream | `|dE| = 2.17e-13` (gate 1e-12; floor 2.18e-13) |
| `get_bands` vs upstream | band **1.681011e-11**, 0/2 bitwise; `mo_energy` **6.102274e-11** (plan 1.68e-11 / 6.10e-11 reproduced); asserted `< 1e-9` (memory band-energies-are-never-bitwise-identical) |
| `krhf_bands_oracle.rs` itself (`PYSCF_ORACLE_VENV=1`, `target/gate`) | exit=0: `delta 2.1715962361668062e-13`, mo 6.102252037010203e-11, band 1.6809886815849495e-11 |
| smearing KRHF He cc-pvdz 2×2×2 σ=0.1 fermi vs upstream `addons.smearing_` | `e_tot` 2.26e-12, `e_free` 2.23e-12, `e_zero` 2.24e-12 (asserted `< 1e-9`); bridged == unbridged bitwise |
| `project_mo_nr2nr` sto-3g→cc-pvdz 3×1×1 vs upstream | **3.22e-15** (asserted `< 1e-9`) |
| KRHF ksymm (3 of 8 k) vs full BZ, He FFTDF | **8.66e-14** (asserted `< 1e-10`); explicit `KsymAdaptedKRHF` bitwise = `KRHF(cell, kpts=KPoints)` |
| KUHF/KROHF/KGHF on closed-shell He vs KRHF | `|dE| = 0`; each bridged == unbridged bitwise; KUHF `<S^2>` 0, bands vs KRHF < 1e-9 |
| chkfile | `load_scf` e_tot/mo_coeff/mo_energy/mo_occ bitwise; h5py reads `scf/mo_coeff` as complex128; `init_guess='chkfile'` restart converges in ≤3 cycles to `< 1e-11`; sto-3g chkfile projected onto cc-pvdz |
| `KUHF(cell, kpts=KPoints(...))` raises; `pyscf.pbc.scf.kuhf_ksymm` not served | yes / yes (`__file__` None, attribute access raises) |
| `grep -n 'phase: 20' crates/pyscf-pbc-scf/src/addons.rs` | no output, exit 1 |
| `cargo test --release -p pyscf-pbc-scf --test project_mo_nr2nr` (`target/gate`, LTO=false) | exit=0, 3 passed |
| `.venv/bin/pytest python/pyscf/tests/test_pbc_override_dispatch.py` | **89 passed** (44 × 2 drivers + 1), exit 0 |
| `.venv/bin/pytest python/pyscf/tests/test_pbc_identity_gate.py` | **12 passed, 6 failed** — scf ×4 now pass; failures dft ×4 (`cannot import name 'radi' from 'pyscf.dft'`), mp, cc (20-13/20-15) |
| `cargo clippy -p pyscf-py --release` | no warning in `pbc/scf.rs` |
| `check-catch-unwind` / `check-forbid-lazy-static` / `check-dependency-wall` / `check-orphan-modules` | exit 0 (844 files) / 0 / 0 (both PASS) / 0 (444 files) |
| full `.venv/bin/pytest python/pyscf/tests -q -p no:cacheprovider` (final `.so`) | **16 failed, 263 passed, 4 skipped, 3 xfailed, 9 errors in 509.74 s** (exit 1, `target/py-20-12-fullsuite.log`). Every PBC file passes except the 6 expected identity-gate cases (dft/mp/cc). The other 10 failures and 9 errors are the known molecular ones from 20-11 D4 / 20-14 (`test_panic_to_exception` ×2, `test_scf_cross_dispatch` ×3, `test_scf_ghf`, `test_scf_rhf_ccpvdz` ×3, `test_scf_rhf_h2o`, plus 9 overlay-import errors in `test_scf_*`), which a separate agent is triaging. Against 20-14 (20 failed, 193 passed): the 4 fixed failures are the scf identity cases, and the +70 passes are `test_pbc_scf` (21) and `test_pbc_override_dispatch`'s KRHF arm. |

## Deviations

- **D1 — the `KPoints` branch is in the native constructors, not only the shim.** The plan puts upstream's `isinstance(arg, KPoints)` dispatch in `pbc/scf/__init__.py`, but the identity gate requires `pyscf.pbc.scf.KRHF is pyscf._native.pbc.scf.KRHF`; a Python dispatch function under that name would break the gate, and a PyO3 `#[new]` cannot return another type. So `KRHF`/`KUHF`/`KGHF` apply the rule at construction (a type check, not MRO); the shim ports the function-shaped names. Consequence: `type(KRHF(cell, kpts=KPoints))` is `KRHF`, not `KsymAdaptedKRHF` (behaviour and `mf.kpts is kp` match upstream).
- **D2 — ksymm cell copy.** Upstream mutates the user's cell (`cell.build_symmetry(kpts)`); here the default FFTDF gets a symmetry-built copy, so `mf.with_df.cell is not mf.cell`. A user-assigned `with_df` over a cell without `symm_orb` fails at `kernel()` with the Rust message (set `use_ao_symmetry = False`). A `build_symmetry` failure (time-reversal `little_cogroup_ops`, 17-07) is deferred to the kernel. The D-17-09-02 `tracing::warn!` in `KsymAdaptedKrhf::kernel` is not emitted on the bridged path.
- **D3 — upstream parity gaps:** gamma shims return per-k lists of length 1 (upstream bare arrays); `smearing()` mutates in place (upstream returns a new object); `density_fit()` swaps `with_df` to `GDF` in place; `kernel(**kwargs)` raises `TypeError`; `get_veff(kpts=/kpts_band=)`, `get_fock(cycle>=0)`, `get_bands` for KROHF/KGHF/ksymm, `init_guess_by_chkfile(kpts=)` and k-point remapping raise `NotImplementedError`; `diis` is a bool; `load_scf` returns a native `Cell`.
- **D4 — `project_mo_nr2nr` solves with pivoted LU** (`zsolve_linear`), upstream with Cholesky; measured 3.22e-15 apart.
- **D5 — the `kuhf_ksymm` stub makes upstream `pyscf.pbc.dft` unimportable once its earlier failure is fixed** (`kuks_ksymm.py` subclasses `kuhf_ksymm.KUHF` at import). Today upstream `pyscf.pbc.dft` already fails first on `cannot import name 'radi' from 'pyscf.dft'`, so nothing observable changed; 20-13 replaces the `pbc.dft` overlay with native KS drivers and must not fall through to upstream `kuks_ksymm`.
- **D6 — build cost.** The `addons.rs` edit re-compiles `pyscf-pbc-scf` dependents (`Finished` 3m00s once); later pyscf-py-only rebuilds 20.87 s. No dependency added to `pyscf-py`.
- **D7 — CubeCL manual (AGENTS.md §3) not consulted:** no compute kernel was written (PyO3 glue plus a host-side `project_mo_nr2nr` over existing integrals/solvers).

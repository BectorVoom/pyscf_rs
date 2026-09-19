# 20-18 PRE-SUMMARY: cheap S shims for the unmodified `examples/pbc` gate

**Shipped:** 2026-09-14. This pre-task covers the six S items of `measurements/example-gate-gap-scope.md` §5, minus the M06 parser fallback, which was skipped by instruction. Three more S gaps surfaced during the example runs and were fixed; they are listed below. One non-S gap was recorded but not fixed.

- Nothing was staged, committed, stashed or restored (D-20-A).
- The `.so` was rebuilt only after `pgrep -f '[p]20-18-suite'` and `pgrep -f '[p]ytest.*pyscf/pbc'` both returned empty. That was 18:16, after the control run finished at 18:10.
- Note: an unbracketed `pgrep -f` pattern matches the polling shell itself. Always use the bracket form.
- Raw logs are in `target/p20-18-examples/`.

## Items

| # | item | where | upstream mirrored |
|---|---|---|---|
| 1 | Single-k-point broadcast in `ao2mo`/`get_eri`. A `(3,)` or `(1,3)` kpt is used for all four indices. | `crates/pyscf-py/src/pbc/df.rs` `PyPeriodicDf::quad` | `_format_kpts`, `pbc/df/fft_ao2mo.py:430-439` |
| 2 | `pyscf.pbc.scf.addons` shim. `smearing_` and `project_mo_nr2nr` are the native objects. Everything else falls through lazily and announced (D-PBC-35). The upstream file imports `pyscf.scf.addons`, so `convert_to_uhf` and similar names raise `AttributeError` naming the cause. | new `python/pyscf/pbc/scf/addons.py`; `_unported.py` scf `missing`/`note` | `pbc/scf/addons.py:28-39` |
| 3 | `pyscf.pbc.dft.multigrid` shim. `MultiGridNumInt` and `MultiGridNumInt2` are native (20-13 names). The submodules are upstream. `pbc/dft/__init__.py` now imports `multigrid`, as upstream does transitively. | new `python/pyscf/pbc/dft/multigrid/__init__.py`; `_unported.py` dft `missing`/`note` | `multigrid/__init__.py:16-17`; `krks.py:33`, `rks.py:39` |
| 4 | `ecc` getter on the native `_KCCSD` base (KRCCSD/KUCCSD/KGCCSD/ksymm) | `crates/pyscf-py/src/pbc/cc.rs` | `cc/ccsd.py:990-992` |
| 5 | `entropy` getter on `KohnShamDFT`, using the same formula as `KSCF.entropy` | `crates/pyscf-py/src/pbc/dft.rs` | `pbc/scf/smearing.py:87-89,123-126` |
| 6a | `pyscf.M(**kw)`: if `a` is not None it returns `pyscf.pbc.gto.M`, otherwise the molecular `gto.M`. `M` was added to `__all__`. | `python/pyscf/__init__.py` | `pyscf/__init__.py:106-112` |
| 6b | Method-style constructors (`cell.KRKS(xc=, kpts=)`, `cell.KRHF()`, `cell.RKS`, `cell.KMP2`, the TD branches). The native `Cell.__getattr__` calls `pyscf.pbc.gto._cell_methods.cell_method`, a line-by-line port. `NotImplemented` means the name is not a method, and the native `AttributeError` stands. `SCF_KW` splitting, the `_MoleLazyCallAdapter` port and the positional-argument refusal are all kept. | `crates/pyscf-py/src/pbc/gto.rs`; new `python/pyscf/pbc/gto/_cell_methods.py` | `pbc/gto/cell.py:1407-1511`, `gto/mole.py:4368-4383` |

### Further S gaps found by the runs (fixed)

| gap | symptom | fix | upstream |
|---|---|---|---|
| **G1** `use_ao_symmetry` was forced on | `22-k_points_mp2_ksymm.py` failed at `kmf.kernel()` with `use_ao_symmetry = true needs cell.symm_orb`. Cause: the binding defaulted to `true`. Upstream ANDs in `not kpts.time_reversal and kpts.symmorphic and len(little_cogroup_ops) > 0`. Diamond Si is non-symmorphic, so upstream takes the plain-`eig` route. | `PyKscf::construct` and `PyKohnShamDft::construct` compute the upstream predicate. They build the `symm_orb` cell copy only when it holds. `KsymAdaptedKRKS/KUKS(use_ao_symmetry=)` now ANDs its argument. | `khf_ksymm.py:142-149` |
| **G2** `sigma` was read-only | `23-smearing.py:47` raised `attribute 'sigma' … is not writable` on KRKS. KRHF had the same problem. | `set_sigma` on `KSCF` and `KohnShamDFT`. A non-zero value updates the smearing width. `0`/`None` drops smearing, as `smearing_(sigma=0)` already does. A non-zero value with no smearing configured raises `AttributeError`. | `scf/smearing.py:130,151,258` |
| **G3** `KohnShamDFT.smearing_method` missing | Found by the new test | Getter, same as `KSCF.smearing_method` | `scf/smearing.py:131` |

G1 also changes which path existing ksymm tests take. `test_pbc_dft`'s non-symmorphic diamond-Si Gate-C cases now run `use_ao_symmetry=False`, as upstream does. The He `symmorphic=True` fixtures are unchanged (`test_pbc_scf` still asserts `use_ao_symmetry is True`). The full suite is green for these tests.

## Tests — `python/pyscf/tests/test_pbc_example_shims.py` (8 cases, 2.3 s)

- **RED:** on the pre-rebuild `.so` 4 of the 7 cases then present failed (`target/p20-18-examples/red-pre-rebuild.log`). The failures were ao2mo `ValueError`, `ecc` `AttributeError`, and two `'Cell' object has no attribute "KRKS"`. The addons, multigrid and `pyscf.M` shims passed on the old `.so`, because they are pure Python.
- **Oracle:** one subprocess on vendored 2.12.1, version asserted.

| gate | measured | bound |
|---|---|---|
| `ao2mo(mo, kpts=k)` with `k`, `k.reshape(1,3)`, `list(k)` vs `kpts=vstack([k]*4)`; the same for `get_eri` | bitwise; two k-points raise `ValueError` | bitwise |
| `ao2mo(mo, kpts=k)` vs upstream (He 6-31g, [1,1,2], non-Γ k, mesh 15) | **8.41e-14** | 1e-10 |
| KRKS PBE smearing through `pyscf.M` + `cell.KRKS` + `mf.smearing` (He cc-pvdz [1,1,2], σ=0.1) | e_tot **1.38e-11**, e_free 1.23e-11, entropy 1.58e-11 | 1e-9 |
| then `mf.sigma = 0.05; kernel()` | e_tot **1.24e-11**, e_free 1.24e-11, entropy 7.5e-14 | 1e-9 |
| `ecc == e_corr` (None before kernel); identity of addons/multigrid; `which_impl` values; `cell.KRKS`/`KRHF`/`KUKS` types, kwargs split, positional refusal, `hasattr` safety | pass | — |

The He fixture is weak for smearing: occupations are nearly integer. The Al example below is the real exercise.

## Example runs (unmodified, `PYTHONPATH=python .venv/bin/python examples/pbc/<x>.py` from the repo root, final `.so` 18:28:19)

Native energies come from each script's own prints. The SCF/`e_corr` values come from a `runpy` wrapper (`run_globals.py`) that only reads the script's globals afterwards. Upstream runs used `run_upstream.py` / `run_globals.py upstream`, which assert `pyscf.__version__ == '2.12.1'` and the vendored `__file__`.

| script | native exit / wall / RSS | upstream exit / wall | quantity | native | upstream 2.12.1 | \|Δ\| |
|---|---|---|---|---|---|---|
| `22-k_points_mp2_ksymm.py` | **0** / 16.4 s / 0.57 GB | 0 / 9.4 s (8.6 s) | KRHF e_tot | -7.52315943854784 | -7.52315941089612 | 2.77e-8 |
| | | | KMP2 e_tot (printed) | **-7.575651916070479** | -7.575653611089853 | **1.70e-6** |
| | | | KMP2 e_corr | -0.05249247752263812 | -0.05249420019373538 | 1.72e-6 |
| `23-smearing.py` | **0** / 26.0 s / 2.56 GB | 0 / 55.0 s (4 OMP threads; 42.0 s default) | Entropy (σ=0.1) | 3.3975718579737046 | 3.382609236505189 | **1.50e-2** |
| | | | Free energy (σ=0.1) | -2.247676980837733 | -2.2426373797484516 | **5.04e-3** |
| | | | ≈zero-T energy (σ=0.1) | -2.0777983879390476 | -2.073506917923192 | 4.29e-3 |
| | | | e_tot after σ=0.001 | -2.0565129955696335 | -2.056513625508845 | 6.30e-7 |
| | | | e_free after σ=0.001 | -2.0569087784633364 | -2.0569087061617015 | 7.23e-8 |

- **ksymm.** Both codes run `use_ao_symmetry=False` (G1). Both have `conv_tol` 1e-7, and the lattice is in Å, so the CODATA-2014 vs CODATA-2010 difference shows in the 8th digit. The KMP2 gap of 1.7e-6 is inside the 2e-6 KMP2 floor (scope doc floor row 7). It was not investigated further.
- **smearing σ=0.1: 5 mHa. RECORDED, not S — M.** Root cause, bisected in scratchpad probes (Al gth-dzvp/gth-pbe, 2×2×2, Bohr lattice at CODATA-2010):
  1. Without smearing, native and upstream agree to 1.8e-10 (2×2×2) and 8.7e-9 (Γ).
  2. The native `get_occ` reproduces upstream `mo_occ` to 8.9e-16 on upstream's `mo_energy`.
  3. Each code returns to its own state from the other's density.
  4. On the SAME density, `hcore`, `veff` and `pp` agree to ≤9.3e-12, and `energy_tot` to 1.8e-9. `S` differs by **2.0e-9**, and Γ's p-like eigenvalue is 0.27807 native vs 0.47372 upstream.
  5. The Γ overlap has a triply degenerate near-null space: λ_min 2.74e-9 at upstream's precision, 4.74e-9 at `cell.precision` 1e-8.
  6. Upstream's SCF `get_ovlp` (`pbc/scf/hf.py:47-55`) integrates at `precision*1e-5`, with `rcut = max(cell.rcut, estimate_rcut(cell, precision*1e-5))` and no `pbcopt` prescreening.
  7. `pyscf_pbc_gto::get_ovlp` (`hcore.rs:46`) ports `scfint.get_ovlp` at plain `cell.precision`. `pbc_intor('int1e_ovlp')` agrees between the codes to 7e-14 at every precision, so the gap is the SCF-level precision bump, not the integrals.
  8. With σ=0.1 the ill-conditioned Γ p band is fractionally occupied (0.77 native vs 0.20 upstream), so the 2e-9 overlap error becomes 5 mHa. At σ=0.001 it is empty, and the gap falls to 6e-7 / 7e-8.
  - **Why M:** the fix is small, but it lives in `pyscf-pbc-gto`/`pyscf-pbc-scf`. That changes the overlap, and therefore the bits, of every periodic SCF. It invalidates printed bitwise references (`krhf_bands_oracle.rs`, `gate.rs`) and needs Rust gate re-runs. It also touches crates carrying uncommitted Phase-18 work.
  - **Carry:** port `pbc/scf/hf.py:get_ovlp`'s precision/rcut rule into the SCF `get_ovlp`, then re-gate.

## Verification

| command | result |
|---|---|
| `cargo check -p pyscf-py` | exit 0; no warnings in touched files |
| `maturin develop --release --skip-install` (EXECUTION-NOTES §2 env) ×3 | exit 0; `Finished` 19.46 s / 11.83 s / 11.74 s (`build{1,2,3}.log`) |
| `rustfmt --edition 2024 --check` on `pbc/{df,cc,dft,gto,scf}.rs` | clean |
| `check-catch-unwind` / `check-dependency-wall` / `check-orphan-modules` | exit 0 (852 files) / 0 (both PASS) / 0 (449 files) |
| `test_pbc_identity_gate.py` | **18 passed** |
| `test_pbc_identity_gate.py test_overlay_resolution.py test_pbc_unported.py` | 68 passed |
| `test_pbc_example_shims.py` | **8 passed** |
| full `.venv/bin/pytest python/pyscf/tests -q -p no:cacheprovider` | **10 failed, 384 passed, 4 skipped, 3 xfailed, 9 errors in 280 s** (`fullsuite.log`). The failing/erroring ids are exactly the known molecular set of `python-suite-triage.md` / 20-17. Zero PBC failures. |

## Deviations

- **D1 — `Cell.__getattr__` skips `from pyscf.pbc import __all__`.** Most of the families it imports fail under the overlay (20-17), and native classes take no registered methods. An `ImportError` from an overlay's lazy fallthrough while probing `getattr(mod, key, None)` counts as "no callable". `pyscf.dft.XC` is imported only on the TD branch. Without the overlay importable, the native `AttributeError` stands.
- **D2 — `cell.KRKS(...)` builds via the overlay `pyscf.pbc.dft`/`scf` names**, so identity holds. `mf.set(**rest)` falls back to a `setattr` loop, because native classes have no `set` (`lib/misc.py:642-661` semantics).
- **D3 — the `sigma` setter** refuses a non-zero value when no smearing is configured (upstream silently stores it). `sigma = 0` drops `smearing_method` (upstream keeps it), so re-enabling needs `smearing_()`.
- **D4 — the single-k `ao2mo` broadcast still requires the k-point to be in `with_df.kpts`**, as the four-k path already did (`kidx`).
- **D5 — upstream timings were taken while the box was otherwise idle.** The 23-smearing upstream run used `OMP_NUM_THREADS=4`; the default-threads re-run took 42.0 s.
- **D6 — CubeCL manual (AGENTS.md §3) not consulted.** No compute kernel was written, only PyO3 glue and Python shims.

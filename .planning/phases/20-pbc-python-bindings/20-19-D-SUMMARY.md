# 20-19 item D: the SCF overlap uses upstream's tightened precision

**Executed:** 2026-09-14. The work was done in an isolated copy (`pyscf_rs-p2019d`) and then copied back. Before copying, each main-tree file was checked against its pristine snapshot, and none had changed. The copy has been deleted.

**Evidence:** `target/p20-19-D/`. It holds the before and after gate logs (trimmed), the non-ignored test logs, the oracle test logs, the maturin log and the example run.

## Upstream rule (vendored 2.12.1, read from source)

`pbc/scf/hf.py:47-55`:

```python
precision = cell.precision * 1e-5
rcut = max(cell.rcut, gto.estimate_rcut(cell, precision))
with lib.temporary_env(cell, rcut=rcut, precision=precision):
    s = cell.pbc_intor('int1e_ovlp', hermi=0, kpts=kpt, pbcopt=lib.c_null_ptr())
```

How each part reaches the integrals:
- **`rcut`** widens `Ls` (`cell.py:222-223`).
- **`precision`** reaches the sum only through the `use_loose_rcut` neighbor list (`rcut_by_shells(precision)`).
- **`hermi=0`** gives the full `s1` fill. No mirroring is done.
- **`pbcopt`** is swallowed by `**kwargs` in 2.12.1, so it has no effect.

### Sites that use this overlap

Upstream sites that use it:
- `SCF.get_ovlp` (`hf.py:645`)
- `khf.get_ovlp` / `KSCF.get_ovlp` (`khf.py:52-63,457`), including the `kpts_band` `s1e` in `get_bands` (`khf.py:690`, `kuhf.py:543`)
- `KsymAdaptedKSCF.get_ovlp` (at `kpts_ibz`)
- `KGHF.get_ovlp` and its `get_init_guess` (`kghf.py:168,227`)
- `krks_ksymm`/`kuks_ksymm`, which alias the ksymm overlap
- `kccsd_rhf_ksymm.py:410`, through `self._scf.get_ovlp()`

Upstream sites that do NOT use it (they call `cell.pbc_intor('int1e_ovlp')` at plain `cell.precision`):
- hcore/`int1e_kin` (`khf.py:88`, `hf.py:643`)
- `krkspu.py`/`kukspu.py`, `scf/addons.py:50`, `df_jk.py:1480`
- the stress `ovlp0` (`krks_stress.py:383`)
- `scfint.get_ovlp`

## Change

| file | change |
|---|---|
| `crates/pyscf-pbc-gto/src/hcore.rs`, `lib.rs` | New `get_ovlp_scf` and `SCF_OVLP_PRECISION_FACTOR`, a line-by-line port of the rule above. Python's `max` keeps the first argument on a tie. The Hermiticity warning is a `tracing::warn`. The DEBUG condition-number warning is not ported. The existing `get_ovlp` (plain `scfint`) is unchanged and documented as not the SCF overlap. |
| `pyscf-pbc-scf/src/{krhf,kuhf,krohf,kghf(×2),khf_ksymm}.rs` | The `get_ovlp` hook and the `get_bands` `s1e` now call `get_ovlp_scf` |
| `pyscf-pbc-dft/src/{krks,kuks,krks_ksymm(×2)}.rs` | Same |
| `pyscf-pbc-cc/src/kccsd_rhf_ksymm.rs:1112` | `get_ovlp_scf`, as upstream's `_scf.get_ovlp()` |
| `pyscf-py/src/pbc/{scf,dft}.rs` | The explicit-`kpts` `get_ovlp(cell, kpts)` route now calls `get_ovlp_scf` |

These callers deliberately keep the plain `get_ovlp`, as upstream does: `kspu.rs`, `addons.rs`, and stress `ovlp0`.

## Tests (separate files, per AGENTS.md §2)

| test | result |
|---|---|
| `pyscf-pbc-gto/tests/scf_ovlp_precision.rs` (oracle-free, 2 tests). `get_ovlp_scf` must be **bitwise** equal to `pbc_intor(hermi=0)` on a clone with `precision*1e-5` and `rcut=max(...)`, on both the default and `use_loose_rcut` routes. Non-vacuity asserts are included. | pass. Al 2×2×2: default route rcut 37.31→44.89, images 1961→3463, \|S_scf−S_plain\| **2.00e-9**. Loose route: images 603→1061, 1.95e-3. |
| `pyscf-pbc-scf/tests/scf_ovlp_oracle.rs` (ignored oracle). The Al `23-smearing` cell, 4×4×4, element level vs `KRHF/KUHF.get_ovlp()` and `get_ovlp(c, kband)`. It asserts the version and that upstream's source still contains the rule. | KRHF hook **2.84e-14**, KUHF 2.84e-14, band-k 3.20e-14 (tol 1e-12). The plain overlap vs upstream is **2.00e-9**; it must exceed 1e-10, and this is the RED state of every driver before the fix. Γ λ_min 2.74e-9. |
| `pyscf-pbc-dft/tests/scf_ovlp_smearing_oracle.rs` (ignored oracle). KRKS PBE, Fermi σ=0.1, Al 4×4×4, default mesh (asserted equal on both sides), conv 1e-10. The tolerance of 1e-7 was fixed before the first measurement. | e_free \|Δ\| **4.82e-9** (was 5.04e-3 in 20-18), e_tot 1.21e-9, entropy 6.0e-8, Γ occ 3.3e-6. 134 s, 2.5 GB. |

**TDD deviation:** `get_ovlp_scf` was written before the gto test file, so that test's RED is a compile failure only. RED for the behaviour comes from two measurements: the oracle test's in-file plain-overlap check (2.0e-9 vs upstream), and 20-18's 5.04e-3 measurement. The smearing gate was not re-run on the pre-fix code.

## T1 gate re-run (same copy)

**BEFORE** means pristine source. The binaries were built, copied aside, and then executed directly. **AFTER** means `run.sh` with the exact CI gate lists.

| crate | before | after (`run.sh`) |
|---|---|---|
| pyscf-pbc-scf | 3/3 pass | **OK 3/3, 0 skips** |
| pyscf-pbc-dft (T1) | 15/15 | **OK 15/15** |
| pyscf-pbc-dft `krhf_si_222_is_the_pseudopotential_floor` (T2) | pass | **OK 1/1** |
| pyscf-pbc-mp | 6/6 | **OK 6/6** |
| pyscf-pbc-df | 14/14 | **OK 14/14** |
| pyscf-pbc-cc | 21/21 | **OK 21/21** |

No gate moved outside its tolerance. Several floors dropped by about 100×:

| gate | before | after | tol |
|---|---|---|---|
| **KRHF Si gth 2×2×2 floor** | **4.158e-12** (tiers doc 4.159e-12) | **2.931e-14** | 1e-11 |
| **KRKS Si PBE 2×2×2** | **6.451e-12** (tiers doc 6.453e-12) | **4.086e-14** | 1e-11 |
| KRKS Si LDA | 6.507e-12 | 3.730e-14 | 1e-11 |
| KUKS Si PBE | 6.448e-12 | 4.086e-14 | 1e-11 |
| KRKS He AE PBE | 8.482e-14 | 6.217e-14 | 1e-12 |
| KRHF / KUHF He AE (kscf) | 2.167e-13 | 7.1e-15 / 6.7e-15 | 1e-12 |
| KRHF bands: mo_energy / get_bands | 6.10e-11 / 1.68e-11 | 3.72e-13 / 4.23e-13 | gate unchanged |
| KUHF Li γ (open-shell floor) | 1.493e-11 | 3.490e-12 | 5e-11 |
| KUKS Li γ PBE / LDA / [1,1,3] / PBE0 | 7.73 / 7.80 / 5.45 / 9.72 e-12 | 3.44 / 3.44 / 3.21 / 3.47 e-12 | 5e-11 |
| KUKS/KUHF H2 γ | 1.75e-13 … 2.58e-13 | 7.6e-14 … 2.57e-13 | 1e-12 |
| KRKS Si PBE vs xcfun (measurement) | 4.709e-7 | 4.709e-7 | — |
| cc / mp / df numbers | — | They changed only in the ≤1e-3 relative digits; the upstream side itself varies run to run at about the 1e-10 level. The mean-field residuals are unchanged (6.91e-6, 1.348e-5). | — |

**Finding:** the "~4e-12 gth-pade oracle floor inherited from `get_pp`" was mostly this overlap-precision gap, not `get_pp`. The same is true of the ~6e-12 KRKS floors and the ~7e-12 Li floors.

## Non-ignored tests (copy, after)

All passed:
- pbc-scf `kscf` 9 (4 ignored), `krhf_bands` 2, `kuhf_bands` 3, `khf_ksymm` 6
- pbc-dft `krks_ksymm` 7 (3 ignored), `smoke` 13, `modules` 8, `kuks_bands` 2
- pbc-cc `kccsd_ksymm` 4
- pbc-gto `hcore` 7, `pbc_intor` 14, `scf_ovlp_precision` 2

## Extension rebuild and `examples/pbc/23-smearing.py`

Before rebuilding, I polled until no `maturin` or `pytest…p20-19-suite` process was running; the rebuild started at 19:51. The EXECUTION-NOTES §2 `maturin develop --release --skip-install` then exited 0 in 30.6 s, and `_native.abi3.so` is stamped 19:52. `test_pbc_example_shims.py` and `test_pbc_identity_gate.py` give 26 passed.

The example was run unmodified with `PYTHONPATH=python .venv/bin/python`: exit 0, 28 s, 2.56 GB.

| quantity | native (after) | upstream 2.12.1 (20-18) | \|Δ\| after | \|Δ\| before (20-18) |
|---|---|---|---|---|
| Entropy (σ=0.1) | 3.3826092138536534 | 3.382609236505189 | **2.3e-8** | 1.50e-2 |
| Free energy (σ=0.1) | -2.2426373775484656 | -2.2426373797484516 | **2.2e-9** | 5.04e-3 |
| ≈zero-T energy (σ=0.1) | -2.073506916855783 | -2.073506917923192 | 1.1e-9 | 4.29e-3 |
| e_tot after σ=0.001 | -2.0565136230905225 | -2.056513625508845 | 2.4e-9 | 6.30e-7 |
| e_free after σ=0.001 | -2.0569087063365643 | -2.0569087061617015 | 1.7e-10 | 7.23e-8 |

The residuals are consistent with the example's `conv_tol` of 1e-7 and its Å lattice (the CODATA-2014 vs CODATA-2010 gap).

## Notes and deviations

- `rustfmt --edition 2024` was applied to the three new test files only. The touched source files were already clean.
- **Not changed, out of scope:** `pyscf-pbc-grad` stress `ovlp0` uses `get_ovlp` with hermi=1, while upstream's `cell.pbc_intor('int1e_ovlp', kpts)` uses hermi=0. That is a bit-level difference only, and the precision is plain on both sides.
- **Stale docs:** the doc comments in `pyscf-pbc-dft/tests/gate.rs` and `measurements/pbc-oracle-tiers.md` still quote the old floors (4.159e-12, 6.453e-12, …). The gate tolerances were not touched.
- The pre-existing `unused import Fftdf` warning at `pyscf-pbc-dft/src/veff.rs:17` does not come from this change.

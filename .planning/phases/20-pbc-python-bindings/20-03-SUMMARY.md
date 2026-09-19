# 20-03 SUMMARY: first CI jobs that exercise periodic code (pbc-oracle / -nightly / -full)

**Shipped:** 2026-09-14. All five tasks done. No commits and no git operations (D-20-A).
No gate tolerance changed. One tolerance was tightened temporarily for Task 5; the
file is byte-identical to its snapshot afterwards (`cmp` exit 0).

**Files**
- `.github/workflows/ci.yml`: added the `schedule` trigger and three jobs (+481 lines).
- `.github/actions/pbc-oracle-setup/action.yml` (new): sibling clones, oracle venv,
  and import-route assertion.
- `.github/actions/pbc-oracle-run/action.yml` + `run.sh` (new): runs one gate set
  with guards against a vacuous pass.

Evidence lives in `target/p20-03/`: run logs, the scratch venvs, `run_push_job.py`,
and `actionlint`/`shellcheck` binaries.

## Headline finding: a stock `pip install numpy` oracle turns two T1 gates red

The harness imports PySCF in one of two ways (from `crates/pyscf-pbc-*/tests`):

| route | who | what `import pyscf` resolves to |
|---|---|---|
| **A** pinned: `cwd=root`, `PYTHONPATH=root` | `common::run_python` (df/dft/scf), cc `emit`, `oracle_kcis`, `oracle_kuccsd`, `oracle_phase15`, `oracle_phase9` | vendored **Python** at `<root>/pyscf` (2.12.1). Its `.so` files come from the **site-packages wheel** through `load_library`'s `__path__` fallback, because the vendored tree has no compiled libraries. |
| **B** unpinned: temp-dir script, no `PYTHONPATH` | gto `gth_pp_loc`, `gth_pp_nl`, `pbc_intor` | site-packages wheel. Locally that is **2.14.0**, so these gates have been measured against 2.14 Python. This is a harness defect, owed to the gto test owner. |

In both routes `PYSCF_ORACLE_VENV=1` resolves to `<root>/.venv/bin/python`, so CI
builds its venv at `$GITHUB_WORKSPACE/.venv`.

**Measured.** The local `.venv` numpy 1.26.4 was built from source on Python 3.13
and **links no BLAS**. A stock numpy wheel links OpenBLAS, and that moves upstream's
own `fft_jk` k_e1 answer on diamond 2×2×2 by **1.712e-7**. The port is unchanged.

| venv (py, pyscf wheel, numpy) | upstream k_e1 vs `.venv` | df T1 set |
|---|---|---|
| `.venv` (3.13, 2.14.0, 1.26.4 no-BLAS) | 0 (re-run: 0; `OMP_NUM_THREADS=1`: 0) | 14/14 PASS |
| venv-d (3.13, 2.14.0, 1.26.4 no-BLAS) | **0** | — |
| venv-f (3.13, 2.14.0, 2.5.3 OpenBLAS) | **1.712e-7** | — |
| venv-b (3.12, 2.12.1, 1.26.4 OpenBLAS) | 1.712e-7 | `e1` FAIL |
| ci-venv (3.12, 2.12.1, 2.5.3 OpenBLAS; the plan's literal `numpy>=1.26`) | 1.712e-7 | **exit 101**: `fft_jk_grad::e1_matches_upstream_gamma_112_222` (1.712e-7 vs 1e-9) and `fftdf::jk_matches_upstream_on_diamond_222` FAIL |
| **venv-e = CI recipe** (3.13, **2.12.1**, 1.26.4 built `-Dblas=none -Dlapack=none`) | last-digit (e-16) | **14/14 PASS**, exit 0 |

venv-f differs from venv-d only in the numpy build (the Python version and native
libs are held fixed), so the BLAS link is the cause. The setup action therefore
installs `numpy==1.26.4 --no-binary numpy -Csetup-args=-Dblas=none
-Csetup-args=-Dlapack=none`, then `pyscf==2.12.1 numpy==1.26.4 scipy==1.17.1
h5py==3.16.0`, all on Python 3.13.

The assertion step checks the following:
- **Route A**: version, `__file__` is the vendored tree, the wheel is 2.12.1,
  `libnp_helper`/`libcgto` load from the wheel, numpy is 1.26.4, and numpy links no BLAS.
- **Route B**: version and site-packages origin.

It was exercised against three venvs:

| venv | exit | why |
|---|---|---|
| venv-e | 0 | — |
| ci-venv | 1 | numpy 2.5.3, BLAS linked |
| venv-d | 1 | wheel 2.14.0 |

The local `.venv` itself would fail this assertion (wheel 2.14.0). CI is stricter
than local.

## Task 1–4: jobs

| job | trigger (`if:`) | shape | gates |
|---|---|---|---|
| `pbc-oracle` | push / pull_request | 1 job, one step per crate (gto, df, scf, dft, mp, cc, ci). Steps use `if: !cancelled() && steps.setup.outcome == 'success'`. Timeout 300 min. | **74** T1 |
| `pbc-oracle-nightly` | schedule (`0 3 * * *`) | matrix: 7 T1 sets + 6 T2 sets + 4 KNOWN-RED rows. Timeout 360 min each. | 130 |
| `pbc-oracle-full` | workflow_dispatch | nightly rows + 9 T3 rows (the heaviest split) + KNOWN-RED diamond gamma. Timeout 360 min each. | 152 |

Shared settings:
- Job env: `PYSCF_ORACLE_VENV=1`, `CARGO_TARGET_DIR=${{ github.workspace }}/target/gate`,
  `CARGO_PROFILE_RELEASE_LTO=false`.
- `rust-cache` uses `workspaces: ". -> target/gate"`.
- Reuse is through the two composite actions plus YAML anchors: `&pbc-oracle-env`,
  the setup and version steps, and every gate set (`&pbc-t1-*`, `&pbc-t2-*`,
  `&pbc-red-*`). GitHub documents anchor support (Context7
  `/websites/github_en_actions`); no merge keys are used.

Every set runs the following (inside `run.sh`):

```
cargo test -p <crate> --release --locked --no-fail-fast --test <t>... -- --ignored --exact --nocapture <gate>...
```

`run.sh` fails in these cases:

| case | exit | how it was proven |
|---|---|---|
| cargo exits non-zero | cargo's code | — |
| Σ"N passed" ≠ number of gates named | 1 | a misspelt gate |
| a skip line is printed while `PYSCF_ORACLE_VENV` is set | 1 | — |
| `CARGO_TARGET_DIR` is under /tmp | 2 | proven |

**Cross-check.** A script compared ci.yml's sets against the
`#[ignore = "Tn:"]` tags in source (157 gates). Every expected gate sits in exactly
one named target of its set, with no duplicates and none missing.

**Exclusions (explicit comments in ci.yml):**
- **Push/PR, 3 T1 unwired arms**, which run as KNOWN-RED in nightly and full:
  - `adc ea::gate_d_ip_spec_factors`: re-measured 9.3579e-1, exit 101
  - `tdscf uhf::live_bigbox_matches_molecular`: panic, exit 101
  - `scf newton_ah::live_newton_matches_upstream_energy`: panic, exit 101
- **All jobs, 5 child entry points (not gates):**
  - gto `eval_ao_point_screen::emit_ao_bits`
  - gto `eval_ao_stages::emit_ao_bits`
  - gto `eval_ao_screen::print_reference_unscreened`
  - gto `eval_ao_image_batch::compare_unscreened_child`
  - tools `fft_thread_child_emits_bits`
- **KNOWN-RED `dft krks_ksymm::kuks_ibz_energy_matches_full_bz`** (T2 fixture
  precondition): runs in nightly and full.
- **`df_ao2mo::get_eri_matches_upstream_on_diamond_gamma`**: tagged T2, but it ran
  2793 s and fails at 3.8427e-8. It is left out of nightly and listed as a
  KNOWN-RED row of `pbc-oracle-full`. **The full job is expected red until that
  carryover closes, and so is the nightly (4 KNOWN-RED rows).** Its ignore tag
  should read T3; that fix is owed to its owner and was not edited here.
- The `df_ao2mo` He-fcc gates are fixed (gdf-ao2mo-regression.md) and are **in** the
  push job.

## Task 5: proof the job can fail (local; the in-CI run is unobserved)

Every run below used `CARGO_TARGET_DIR=$PWD/target/gate CARGO_PROFILE_RELEASE_LTO=false`
and the `pyscf-pbc-df` T1 set: 7 targets, 14 gates, via `run.sh`, which is the exact
job command.

| # | setting | exit | result |
|---|---|---|---|
| 1 | `PYSCF_ORACLE_VENV=1` (`.venv`) | **0** | 14/14, 0 skip lines, 39 s |
| 2 | `PYSCF_ORACLE_VENV` unset | **0** | 14/14 ok, 13 skip lines (skip contract holds). Plain cargo line on 4 targets: exit 0. |
| 3 | tightened `df_ao2mo.rs:921` `w < 1e-11` → `1e-12` (measured 1.6667e-12), `PYSCF_ORACLE_VENV=1` | **101** | `get_eri_matches_upstream_on_he_fcc` panicked, `::error … exited 101` |
| — | reverted via Edit; `cmp` against the pre-edit snapshot (sha256 `d4c69b45…`) | cmp **0** | byte-identical |
| 4 | CI-recipe venv-e | **0** | 14/14 |
| 5 | stock-wheel ci-venv | **101** | 2 FAIL (the numpy/BLAS finding above) |

**Whole push job, replayed locally.** `run_push_job.py` reads the `pbc-oracle` steps
from ci.yml and runs `run.sh` for each. Settings: venv-e, `RUST_TEST_THREADS=4`
(the runner's nproc), `MemoryMax=16G`.

| crate | gates | result | seconds (incl. builds) |
|---|---:|---|---:|
| gto | 14 | ok | 35 |
| df | 14 | ok | 173 |
| scf | 3 | ok | 57 |
| dft | 15 | ok | 371 |
| mp | 6 | ok | 36 |
| cc | 21 | ok | 362 |
| ci | 1 | ok | 7 |

**ALL GREEN, driver exit 0: 74/74 T1 gates.**

**Not observed:** the jobs have never run on GitHub. The first push must be watched
for the following:
1. The numpy source build on the runner.
2. `--locked` against freshly cloned sibling HEADs. Locally `rmath` has 10
   uncommitted changes and `xcfun_rs` 642.
3. Wall time and disk on 4-vCPU / 16 GB runners.
4. Numpy SIMD dispatch on runner CPUs: the local host is a Ryzen AI 7 350 with
   AVX-512; runners are usually EPYC.
5. Nightly/full T3 rows possibly hitting 360 min.

## Deviations (EXECUTION-NOTES wins on facts)

- **Venv recipe.** The plan's `"numpy>=1.26" "scipy>=1.11" "h5py>=3.10"` floats to
  OpenBLAS numpy 2.5.3, which is measured red (see above). Exact pins and a
  BLAS-less numpy build replace them. Python is 3.13, not the system interpreter.
- **Target dir.** `target/gate`, not `$HOME/.cargo-target-gate` (local protocol spelling, rust-cache).
- **Sibling crates.** `setup-sibling-crates` does not clone `cube-math` (used by
  `pyscf-kernels`) or `rmath` (used by `pyscf-pbc-gto`), so the setup action clones
  both. **Out-of-scope finding:** every existing Rust job in ci.yml has lacked these
  path deps since 2026-08-27 (unobserved; it cannot be checked from here). Moving
  the clones into `setup-sibling-crates` would fix those jobs.
- **Schedule side effect.** Adding `schedule` to ci.yml also runs every job without
  an `if:` nightly.
- **Extra guard.** `--no-fail-fast` was added so one failing binary does not hide the others.
- **Test threads.** CI uses cargo's default. Local runs within one binary were in
  parallel at 4 threads and stayed green.

## Verification

```text
grep -c pbc .github/workflows/ci.yml            -> 124   (HEAD: 0)
grep -n 'pyscf==2.12.1' ci.yml                  -> :916 job name, :934 pyscf-requirement (+ 6 pre-existing)
grep -rn -- '--tests' ci.yml .github/actions/pbc-oracle-*  -> no hits
pyyaml safe_load ci.yml + 3 action.yml          -> OK
actionlint 1.7.12 (+shellcheck 0.10.0) ci.yml   -> 1 finding, pre-existing (:435 `if: false`, also on HEAD)
shellcheck run.sh + composite run blocks        -> clean
cmp df_ao2mo.rs snapshot                        -> 0
```

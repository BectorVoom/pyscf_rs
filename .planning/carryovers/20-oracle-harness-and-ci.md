# Carryover — oracle harness and CI defects found in Phase 20

**Source:** `.planning/phases/20-pbc-python-bindings/20-03-SUMMARY.md`, `20-17-SUMMARY.md` D1,
`measurements/upstream-pbc-suite.md` §Harness finding; recorded by 20-18 (`20-VERIFICATION.md`).

## (i) gto oracle gates compare against pyscf 2.14.0 locally

`crates/pyscf-pbc-gto/tests/{gth_pp_loc.rs:407-416, gth_pp_nl.rs:~294, pbc_intor.rs:~1033}` write
the oracle script to `std::env::temp_dir()` and run it with `current_dir(workspace_root())` but **no
`PYTHONPATH`**. A script's own directory — not the cwd — is `sys.path[0]`, so `import pyscf`
resolves to `.venv` site-packages, which is **2.14.0** locally (2.12.1 only in the CI venv, where
the setup action asserts the route). These 14 T1 gates (`g_space_factors_*`, `vnl_*`, `ovlp_*`,
`kin_*`) have therefore been measured against 2.14 Python on this box. Fix: pin
`PYTHONPATH=<root>` and assert `pyscf.__version__ == "2.12.1"` like `common::run_python` (df/dft/scf).

## (m) CI

1. **`schedule` trigger side effect.** 20-03 added `schedule: 0 3 * * *` to `ci.yml`; every job
   without an `if:` guard now also runs nightly.
2. **`setup-sibling-crates` does not clone `cube-math` (used by `pyscf-kernels`) or `rmath` (used by
   `pyscf-pbc-gto`).** The `pbc-oracle-setup` action clones both, but every other Rust job in
   `ci.yml` has lacked these path deps since 2026-08-27 (unobserved). This includes the
   `oracle-determinism` job, which is why 20-18's periodic determinism step lives in the
   `pbc-oracle` job, not in that matrix.
3. **First-push CI run unobserved.** The `pbc-oracle*` jobs have never run on GitHub. Watch:
   numpy 1.26.4 source build (`-Dblas=none`) on the runner; `--locked` against fresh sibling HEADs
   (locally `rmath` has 10 uncommitted changes, `xcfun_rs` 642); wall time/disk on 4-vCPU/16 GB;
   numpy SIMD dispatch on EPYC vs local AVX-512; nightly/full T3 rows vs the 360-min limit; the
   20-18 `PBC determinism` step.
4. **Nightly and full are expected red** (4 KNOWN-RED rows + `df_ao2mo` diamond gamma).

## Python harness

- `pytest.ini` `--import-mode=importlib` imports vendored `pyscf/pbc/__init__.py` for upstream test
  files even with the overlay first on `PYTHONPATH` (0 collected, 111/111 collection errors). Any
  CI job for the upstream suite must pass `--import-mode=prepend` after `-c pytest.ini`.
- `.venv` without `PYTHONPATH` imports site-packages 2.14.0 (`pyscf_rs.pth` appends `python/` after
  site-packages); from the repo root cwd `''` imports vendored 2.12.1. Examples must be run as
  `PYTHONPATH=$REPO/python .venv/bin/python …` (`docs/pbc-status.md §2`).

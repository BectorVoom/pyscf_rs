# REPORT-T17 — T4 validation (OWNER ACTION REQUIRED)

## Prepared (implementation agent, 2026-10-02)

- CUDA runner built via podman (`tools/kaggle-t4/build_runner.sh`,
  `BUNDLE=/home/user/Documents/workspace/.yta_bundle`), `Finished release
  profile in 15m 59s`.
- Runner SHA-256:
  `a65d0ce43e392c916a39075a693adcda91990b858e7c37e4afa4cfcb7a960ed8`
  (`/home/user/Documents/workspace/.yta_bundle/runner/yta7o19_bands`).
- Run configuration:
  `/home/user/Documents/workspace/.yta_bundle/ktrun/ktao3-t4.json`
  (`run_name: ktao3-parity-t4`, KTaO3 cell text embedded in `YTA_CELL`,
  `YTA_KMESH 3,3,3`, `YTA_NPATH 16`, `YTA_CONV 1e-6`, budgets
  `NUMINT_AO/FFTJK_AO 256 MB`, `INTOR_IMAGE_CACHE 1 MB`, `PP_NL 16 MB`,
  `YTA_REQUIRE_BACKEND/PYSCF_BACKEND cuda`). `kaggle.user` and
  `kaggle.token_file` left EMPTY for the owner.
- Config check: cell text round-trips
  (`YTA_CELL` → `KTaO3 (cubic perovskite, a = 3.9885 A)`).

## Local reference (T16, CPU backend, same recipe)

- `s2 d_e_tot 4.98e-10` (< 1e-7), converged T/T, cycles 3/3.
- Bands `3.37e-6` (< 1e-5), `d_gap 8.60e-6 eV` (< 1e-4).
- `pre`/`s1` cycles 12/13, 13/14 (within ±1 of upstream).

## OWNER ACTION (do not run as the agent)

```bash
cd /home/user/Documents/workspace/pyscf_rs/tools/kaggle-t4
./ktrun.py publish-runner /home/user/Documents/workspace/.yta_bundle/ktrun/ktao3-t4.json
./ktrun.py run            /home/user/Documents/workspace/.yta_bundle/ktrun/ktao3-t4.json
# when it reports DONE:
PYTHONPATH=/home/user/Documents/workspace/pyscf_rs /home/user/Documents/workspace/pyscf_rs/.venv/bin/python \
    upstream_check.py <local_dir>/ktao3-parity-t4/session_1/out/ckpt
```

Acceptance: the T16 limits, and `backend` in `result.json` is `cuda`.
Also record from the session log: the `h1e.bin: computed in …` time and
the peak GPU memory.

## Update (review, 2026-10-03)

The runner was rebuilt after the review fixes (smearing gradient, host fold).
The SHA above is stale. Current runner:
`4e9ab7aacd1df75ed74ee740a111db9260824941c5fdb4c40dbeaa7e5f53cbe9`
(`/home/user/Documents/workspace/.yta_bundle/runner/yta7o19_bands`).
The configuration file and the owner commands are unchanged. Acceptance uses
the limits of T16 as revised (1e-6 run: bands < 5e-5 Ha, gap < 2e-3 eV;
`backend` must be `cuda`).

## T17 DONE (2026-10-03, boomvector, Tesla T4)

Runner `2c9e215bb179be2d657f38d5226f58a8586737aac09f23bde2824147a406571a`
(D1–D6), run `ktao3-parity-t4b`, `backend = cuda`, 0.12 h, peak 2.8 GB GPU.
Against STOCK PySCF 2.12.1 (`upstream_check.py`, CPU): every SCF cycle of
pre/s1/s2 within 7.4e-13 Ha, cycle counts 13/14/3 identical, bands 4.3e-11
Ha, gap 6.5e-13 eV.

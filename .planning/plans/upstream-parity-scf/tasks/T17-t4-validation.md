# T17 — T4 validation (the owner starts the run)

**Goal.** The same KTaO3 comparison with the pipeline running on a Kaggle T4
(CUDA backend). This is the first time the new kernels run on a GPU.

**You prepare; the owner decides.** Steps 1–3 are yours. Step 4 uses the
owner's Kaggle account and GPU quota: do NOT run it. Write the commands into
`tasks/REPORT-T17.md` and stop.

## Do

1. Build the CUDA runner (podman; no GPU needed locally; long):

```bash
cd /home/user/Documents/workspace/pyscf_rs/tools/kaggle-t4
BUNDLE=/home/user/Documents/workspace/.yta_bundle ./build_runner.sh 2>&1 | tail -5
sha256sum /home/user/Documents/workspace/.yta_bundle/runner/yta7o19_bands
```
2. Write the run configuration
   `/home/user/Documents/workspace/.yta_bundle/ktrun/ktao3-t4.json`: copy
   `tools/kaggle-t4/config.example.json`, then set
   - `"run_name": "ktao3-parity-t4"`
   - in `env`: `"YTA_CELL"` = the CONTENT of
     `/home/user/Documents/workspace/.yta_bundle/upcheck/ktao3.json` as one
     JSON string (T15 step 5 made the runner accept the text itself),
     `"YTA_KMESH": "3,3,3"`,
     `"YTA_NPATH": "16"`, `"YTA_CONV": "1e-6"`,
     `"PYSCF_PBC_NUMINT_AO_BUDGET_MB": "256"`,
     `"PYSCF_PBC_FFTJK_AO_BUDGET_MB": "256"`,
     `"PYSCF_PBC_INTOR_IMAGE_CACHE_MB": "1"`,
     `"PYSCF_PBC_PP_NL_BUDGET_MB": "16"`,
     `"YTA_REQUIRE_BACKEND": "cuda"`, `"PYSCF_BACKEND": "cuda"`.
   Leave `kaggle.user` and `kaggle.token_file` EMPTY — the owner fills them.
3. Check the configuration parses and the cell text survives:
   `python3 -c "import json;c=json.load(open('/home/user/Documents/workspace/.yta_bundle/ktrun/ktao3-t4.json'));print(json.loads(c['env']['YTA_CELL'])['name'])"`
   prints the KTaO3 name.
4. **OWNER ACTION** (write these into `REPORT-T17.md`, do not run):

```bash
cd /home/user/Documents/workspace/pyscf_rs/tools/kaggle-t4
./ktrun.py publish-runner /home/user/Documents/workspace/.yta_bundle/ktrun/ktao3-t4.json
./ktrun.py run            /home/user/Documents/workspace/.yta_bundle/ktrun/ktao3-t4.json
# when it reports DONE:
PYTHONPATH=/home/user/Documents/workspace/pyscf_rs /home/user/Documents/workspace/pyscf_rs/.venv/bin/python \
    upstream_check.py <local_dir>/ktao3-parity-t4/session_1/out/ckpt
```
   Acceptance: the limits of T16, and `backend` in `result.json` is `cuda`.
   Also record from the session log: the `h1e.bin: computed in …` time and
   the peak GPU memory.

## Verify

`REPORT-T17.md` exists and contains: the runner SHA-256, the configuration
path, and the owner commands.

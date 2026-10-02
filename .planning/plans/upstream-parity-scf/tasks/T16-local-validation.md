# T16 — Local validation against stock PySCF

**Goal.** The whole pipeline, with the three fixes, agrees with stock
upstream PySCF on KTaO3. No source changes.

## Do

1. Run the pipeline (CPU backend, small budgets so every batched path runs;
   about 5–10 minutes):

```bash
cd /home/user/Documents/workspace/pyscf_rs
U=/home/user/Documents/workspace/.yta_bundle/upcheck; rm -rf $U/t16; mkdir -p $U/t16
env YTA_CELL=$U/ktao3.json YTA_CKPT_DIR=$U/t16 YTA_REQUIRE_XC=Libxc \
    YTA_BASIS=gth-dzvp-molopt-sr YTA_PRE_BASIS=gth-szv-molopt-sr YTA_PRE_CONV=1e-4 \
    YTA_KE=60 YTA_KMESH=3,3,3 YTA_NPATH=16 YTA_CHUNK=8 YTA_CONV=1e-6 YTA_MAXCYC=80 \
    YTA_REQUIRE_CONVERGED=1 YTA_SMEARING=gauss:0.03 YTA_DAMP=0.7 YTA_LEVEL_SHIFT=0.15 \
    YTA_DIIS_START=8 YTA_DIIS_SPACE=16 YTA_REFINE_SMEARING=gauss:0.002 \
    YTA_REFINE_LEVEL_SHIFT=0.05 PYSCF_PBC_INTOR_PAIR_BATCH=1 \
    PYSCF_PBC_NUMINT_AO_BUDGET_MB=256 PYSCF_PBC_FFTJK_AO_BUDGET_MB=256 \
    PYSCF_PBC_INTOR_IMAGE_CACHE_MB=1 PYSCF_PBC_PP_NL_BUDGET_MB=16 \
    target/release/examples/yta7o19_bands > $U/t16/run.log 2>&1; echo exit=$?
```
2. Compare with stock PySCF (about 5 minutes):

```bash
PYTHONPATH=$PWD .venv/bin/python tools/kaggle-t4/upstream_check.py $U/t16 > $U/t16/upstream_check.log 2>&1; echo exit=$?
python3 - <<'PY'
import json
r = json.load(open('/home/user/Documents/workspace/.yta_bundle/upcheck/t16/upstream_check.json'))['stock']
for st in ('pre', 's1', 's2'):
    x = r[st]; print(st, 'cycles', x['cycles_port'], x['cycles_upstream'], 'converged', x['converged_port'], x['converged_upstream'], 'dE %.2e' % x['d_e_tot'])
b = r['bands']; print('bands max %.2e  gap port %.6f upstream %.6f  d_gap %.2e eV' % (b['d_all_max'], b['port']['gap_ev'], b['upstream']['gap_ev'], b['d_gap_ev']))
PY
```

## Verify — every line must hold

At the pipeline tolerance (`YTA_CONV=1e-6`) the two codes stop at slightly
different densities, so band energies agree only to the SCF noise:

| quantity | limit |
|---|---|
| step 1 `exit` | `0` |
| cycle counts `pre` / `s1` / `s2` | equal to upstream's |
| `s2`: `d_e_tot` | `< 1e-7` Ha |
| `s2`: `converged` port and upstream | both `True` |
| bands `d_all_max` | `< 5e-5` Ha |
| `d_gap_ev` | `< 2e-3` eV |

Then repeat steps 1–2 in a new directory (`$U/t16tight`) with `YTA_CONV=1e-10`
and `YTA_MAXCYC=120`. With both codes converged tightly the limits are the
real parity test:

| quantity | limit |
|---|---|
| cycle counts | equal to upstream's |
| `s2`: `d_e_tot` | `< 1e-10` Ha |
| bands `d_all_max` | `< 1e-6` Ha |
| `d_gap_ev` | `< 1e-5` eV |

Measured 2026-10-03 (review): at 1e-6 — cycles 13/13, 14/14, 3/3, `s2`
1.3e-8 Ha, bands 2.0e-5 Ha, gap 6.6e-4 eV; at 1e-10 — cycles 13/13, 23/23,
3/3, `s2` 2.8e-14 Ha, bands 6.1e-8 Ha, gap 2.1e-6 eV.

Copy the printed lines into `PROGRESS.md`.

## If it fails

- Step 1 `exit=3`, `4` or `5` (a stage did not converge): rerun with
  `YTA_MAXCYC=150`. If it still fails, STOP and report the last 20
  `periodic SCF cycle` lines of `run.log`.
- A limit is exceeded: STOP and report all printed lines. Do not change a
  tolerance. For reference, before the fixes the stock comparison gave
  `d_e_tot = 1.9e-3`, bands `3.5e-3`, `d_gap = 3.3e-3`.

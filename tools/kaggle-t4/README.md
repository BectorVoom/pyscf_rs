# Resumable band-structure runs on Kaggle T4 GPUs

A large periodic DFT band structure does not fit one Kaggle session. For
example, YTa7O19 with `gth-dzvp-molopt-sr`, a 60 Ha cutoff and a 3×3×1 k-mesh
takes about 55 minutes per SCF cycle on a T4, and a GPU session ends after
12 hours. This tool splits the run into sessions. Each session picks up where
the last one stopped. You start it with one command and can walk away.

## How it works

```
 your machine                                   Kaggle
 ───────────────────────────────────────        ──────────────────────────────────────────
 ktrun.py run config.json
   ├─ push session N ─────────────────────────▶ notebook: checks GPU, runner SHA, checkpoint
   │                                             runs yta7o19_bands until done or ~11 h
   │                                             writes ckpt/ after every stage and SCF cycle
   ├─ wait (polls every 5 min)
   ├─ pull session N, verify checksums ◀──────── /kaggle/working/ckpt
   ├─ print progress (stage, cycles, energies)
   └─ publish ckpt as dataset <user>/<run>-ckpt-sN ─▶ input of session N+1
```

The binary runs these stages. A finished stage is skipped on the next start.

| stage | what | resumes at |
|---|---|---|
| `1e` | overlap + core Hamiltonian (`s1e.bin`, `h1e.bin`) | the missing matrix |
| `pre` | optional SCF in a small basis on the same grid and k-mesh (`YTA_PRE_BASIS`) | the last finished cycle |
| `project` | projects the `pre` density into the target basis (`dm0.bin`) | reruns in seconds |
| `s1` | target-basis SCF with broad smearing | the last finished cycle |
| `s2` | re-converge with narrow smearing (`YTA_REFINE_SMEARING`) | the last finished cycle |
| `bands` | band energies along the k-path, `YTA_CHUNK` points per call | the last finished chunk |

Each SCF cycle saves one checkpoint file: the density, the Fock matrix it
used, the cycle number, whether that cycle converged, and the smearing and
tolerance the stage converges to. A stage counts as done only when that
file says it converged with the settings you are running now. If you change a
stage's smearing, it restarts from its saved density. Anything built on it
(the next SCF stage and the bands) is recomputed. `fingerprint.json` records
basis, pre-basis, cutoff, k-mesh and XC, and a checkpoint from a different
calculation is refused rather than mixed in. A resumed SCF therefore keeps damping from its
first step. Before DIIS starts, it repeats the uninterrupted run bit for bit
(`crates/pyscf-pbc-scf/tests/kscf_resume.rs`). After DIIS has started, the DIIS
history restarts empty.

## One-time setup

1. **Kaggle token.** In Kaggle, open *Settings → API → Create New Token*.
   Save the token in a file (for example `~/.kaggle/access_token`) with mode
   `600`. Install the CLI with `pip install kaggle`. Your account must have GPU
   quota; a T4 gives about 30 h per week.
2. **Runner binary.** Build it once per code change:
   ```bash
   BUNDLE=~/Documents/workspace/.yta_bundle tools/kaggle-t4/build_runner.sh
   ```
   This builds in `ubuntu:22.04` so the binary matches Kaggle's glibc. It
   needs `podman` and the build bundle; see the comment at the top of the
   script.
3. **Config.** Copy `config.example.json` and fill in `kaggle.user`,
   `kaggle.token_file`, `runner.binary` and a `run_name` (6–32 characters).
   The `env` block holds the physics and the T4 memory budgets:
   - The example config is YTa7O19 with DZVP, ke 60 and 3×3×1, pre-converged
     in SZV.
   - Memory budgets: XC blocks 2.5 GB, Coulomb blocks 2 GB, band blocks 1 GB,
     and the device AO cache off. That fits a 15 GB T4.

## Running

```bash
cd tools/kaggle-t4
./ktrun.py publish-runner my-run.json   # uploads the binary (once per build)
./ktrun.py run my-run.json              # sessions until done; Ctrl-C is safe
./ktrun.py status my-run.json           # where is it? (any time, any terminal)
```

- `run` can be stopped and restarted at any time. It first waits for the
  session it already pushed. If a push never reached Kaggle, `run` notices
  from the output's session number and pushes again. Once the run is done,
  `run` does nothing.
- `run -n 2` stops after two sessions, which helps you ration the weekly quota.
- `seed-ckpt my-run.json DIR` starts session 1 from files you already have.
  For example, `s1e.bin` and `h1e.bin` from an earlier run with the same cell,
  basis, cutoff and k-mesh skip the one-electron stage. Files without a
  `fingerprint.json` are adopted explicitly: session 1 runs with
  `YTA_ADOPT_CHECKPOINT=1`. Only seed files that really come from the same
  calculation.
- Everything pulled is kept in `~/ktrun/<run_name>/session_N/out/`. The
  results are in `ckpt/result.json`, the plot is in `bands.png`, and the full
  log is in `run.log`.

`status` output looks like:

```
run yta7o19-dzvp-ke60-k331: 3 session(s) finished
  stage: s1   {"cycle": 21, "e_tot": -1505.1234567}
   pre:  25 cycles, last e_tot: -1503.39...
    s1:  22 cycles, last e_tot: ...
        ~55 min per cycle
```

## When it stops by itself

| message | meaning | what to do |
|---|---|---|
| `wrong machine for t4` | Kaggle gave a different GPU | `run` again |
| `stale runner mounted` | Kaggle mounted an old dataset version | `publish-runner` again, then `run` |
| `an SCF stage hit YTA_MAXCYC unconverged` | the SCF did not converge | change the smearing, damping or level shift in `env`, raise `YTA_MAXCYC`, and `run` again. The stage resumes from its last cycle with the new settings. |
| `this session made no progress` | the binary failed early | read `session_N/out/run.log` |
| `no ckpt_manifest.json` | the session was killed before saving | `run` again; it restarts from the previous checkpoint |
| `checkpoint fingerprint mismatch` (in `run.log`) | basis, cutoff, k-mesh or XC changed | use a new `run_name` for a new calculation |

Each session's checkpoint is its own private dataset
(`<user>/<run_name>-ckpt-sN`). You can delete old ones on kaggle.com once the
run is done.

## Tests

- `python3 -m unittest discover tools/kaggle-t4/tests` runs the driver
  against a fake `kaggle` CLI. It covers:
  - a normal run to the end;
  - a driver killed after Kaggle accepted a push;
  - a push that never landed;
  - the no-progress stop;
  - a failed stage;
  - a torn history line;
  - the wrong account.
- The resume guarantee is tested in Rust by `crates/pyscf-pbc-scf/tests/kscf_resume.rs`.
- The small-basis → large-basis projection is tested by
  `crates/pyscf-pbc-dft/tests/krks_dzvp_small_cell.rs`.

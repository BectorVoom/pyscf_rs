# T01 — Baseline

**Goal.** Record what passes today, so later tasks can tell their own changes
from old failures. No source file changes.

## Do

1. Run these four commands one after another (TEST form from `README.md`).
   Save the last 30 lines of each output.

   | crate | targets |
   |---|---|
   | `pyscf-kernels` | `--test pbc_gv --test pbc_band_vmat` |
   | `pyscf-pbc-scf` | `--test kscf --test kscf_resume --test grad_fc_hoist` |
   | `pyscf-pbc-df` | `--test fftdf` |
   | `pyscf-pbc-dft` | `--test krks_dzvp_small_cell --test numint_ao_budget` |

2. Run the two oracle targets that exist today (ORACLE form):
   - `-p pyscf-pbc-df --test fftdf`
   - `-p pyscf-pbc-dft --test scf_ovlp_smearing_oracle`
3. Write `.planning/plans/upstream-parity-scf/baseline.txt`: for each
   command, the command line, then every line that starts with `test ` or
   `test result:`.
4. Copy the current runner so later tasks can compare against it:
   `cp target/release/examples/yta7o19_bands /home/user/Documents/workspace/.yta_bundle/upcheck/yta7o19_bands.before`

## Verify

`grep -c "test result:" .planning/plans/upstream-parity-scf/baseline.txt`
prints `10` (one line per test target: 8 from step 1, 2 from step 2).

## If it fails

- A test fails here: do NOT fix it. Record it in `baseline.txt` under a line
  `KNOWN FAILURE BEFORE THIS PLAN:` and continue. Later tasks must not make
  that list longer.
- A build is killed (out of memory): rerun the same command with `-j8`
  added after `--release`.

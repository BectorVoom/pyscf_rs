# REPORT-T07 — Final diagonalisation: the band pipeline (BLOCKED, STOP rule)

## Verdict

T07 Verify step 2 fails: neither SCF stage converges within 80 cycles, so the
final diagonalisation never runs (`"final":true` count 0, `final
diagonalisation` log count 0; expected 1 each). Two attempts made (one
contaminated by a concurrent session — see §4 — one clean solo re-run).
Upstream PySCF 2.12.1 run on the identical recipe ALSO fails to converge (and
diverges), so this is recipe stiffness on this cell, not a port regression.
Per README rule 9: STOP. Do not continue to T08. Do not weaken a tolerance.

## Attempt 1 (contaminated — see §4)

```bash
U=/home/user/Documents/workspace/.yta_bundle/upcheck; rm -rf $U/t07; mkdir -p $U/t07
env YTA_CELL=$U/ktao3.json YTA_CKPT_DIR=$U/t07 YTA_BASIS=gth-szv-molopt-sr YTA_KE=60 \
    YTA_KMESH=3,3,1 YTA_NPATH=8 YTA_CONV=1e-7 YTA_SMEARING=gauss:0.03 YTA_DAMP=0.7 \
    YTA_LEVEL_SHIFT=0.15 YTA_DIIS_START=8 YTA_DIIS_SPACE=16 \
    YTA_REFINE_SMEARING=gauss:0.002 YTA_REFINE_LEVEL_SHIFT=0.05 \
    target/release/examples/yta7o19_bands 2> $U/t07/run.log; echo exit=$?
grep -c '"final":true' $U/t07/history.jsonl
grep -c "periodic SCF final diagonalisation" $U/t07/run.log
```

Result: `exit=0`; both greps `0` (expected 1). Stages: s1 80 cycles
unconverged (e −134.4512165829), s2 80 cycles unconverged (e −134.4520885107).

## Attempt 2 (clean solo dir, private name to avoid the concurrent session)

Dir `$U/t07solo_opencode`, same env, `rm -rf` + verified empty before launch.
Result: `exit=0`; both greps `0`. History exactly 80 s1 + 80 s2 entries, all
`converged: false`.

```
[yta] stage s1 converged=false cycles=80 e_tot=-134.4509081002 Ha  in 49.4s
[yta] stage s2 converged=false cycles=80 e_tot=-134.4520889472 Ha  in 52.2s
```

s1 tail |dE| ≈ 3–7e-5/cycle, s2 tail |dE| ≈ 2e-6/cycle with norm_gorb ≈ 5.4e-3
(needs |dE| < 1e-7 AND grad < sqrt(1e-7) ≈ 3.2e-4). Monotonic crawl, hundreds
of cycles needed at best. A `YTA_MAXCYC=150` run (in the contaminated window)
also did not converge (s2 at 150: e −134.45218607835747, still falling
~1e-6/cycle).

## Upstream control (same recipe, vendored PySCF 2.12.1)

`/tmp/opencode/probe_s1.py`: KTaO3, `gth-szv-molopt-sr`, `gth-pbe`,
`ke_cutoff` 60 (mesh [29,29,29], same as port), k-mesh 3×3×1, KRKS/PBE,
gauss:0.03, damp 0.7, shift 0.15, DIIS start 8 space 16, conv 1e-7, max 80:

```
CONVERGED False cycles 80 e -134.43336525456698
```

Upstream stalls near cycle ~60 (|dE| ~3e-7, |g| ~2e-3) then DIVERGES
(|g| → 0.11, |ddm| → 0.44 by cycle 80). So the T07 recipe does not converge
upstream either. The port (monotonic to −134.451, still falling) is healthier
here; energies differ because the MINAO guesses differ (known separate item).

## Interference note (another session on this machine)

During this work a second operator ran pipelines concurrently: unexpected
dirs `$U/t07before`, `$U/t07m2` appeared, `yta7o19_bands.before` was
re-copied (19:55, after my 19:44 copy), a live
`target/release/examples/yta7o19_bands` process (PID 2475834) overlapped my
runs, and Attempt 1's `history.jsonl` holds doubled entries (s1 80+68, s2
80+80) plus a bogus "resuming at cycle 12". Attempt 2 used a private dir and
is clean (80+80 exactly). Recommend coordinating checkpoint dirs.

## What was verified working anyway (T07 mechanism, minus convergence)

- `cargo build --release -p pyscf-pbc-dft --example yta7o19_bands` → Finished.
- Second run in the solo dir: `resuming at cycle 80` for both stages, no
  `controls changed` restart, `exit=0`, `result.json` gap stable
  (0.0284 eV). Identity strings are stable across sessions.
- The final-diagonalisation mechanism itself is proven by T05/T06
  (`kscf_conv_check_oracle` green, Γ mo_energy |Δ| 2.7e-8 vs stock).

## Tried (within the task's allowance)

- Exact task command (Attempt 1) and clean solo re-run (Attempt 2).
- `YTA_MAXCYC=150` (T16-blessed): still unconverged.
- No tolerances weakened, no tests deleted/ignored, no recipe constants
  changed (that would exceed this task).

## Suggested remedies (owner/author decision)

1. Retune the KTaO3 verify recipe (stiffer shift/smearing/DIIS or more
   cycles) — note upstream itself diverges here, so any recipe must be
   validated against upstream first.
2. Or validate the T07 mechanism on an easy cell (diamond 2×2×2 PBE shift 0.2
   converges in 9 cycles per T05) instead of KTaO3 for this task, keeping
   KTaO3 for T16.

## Addendum — second session (implementation agent, 2026-10-02 ~21:05)

A second session is executing the same plan in this tree (owner instruction:
"proceed plan until end"). From its side, THIS report's author is the
concurrent operator. Corroborating, independently obtained evidence:

- BEFORE binary + trajectory-equivalent recipe (old shift 0.3/0.1) fails
  identically (`s1 80 unconverged e −134.4509415604`, `s2 80 unconverged
  e −134.4520910007`).
- T03 half-shift identity on a real SCF (new@0.15 vs old@0.3, stage s1):
  cycles 0–2 bit-identical (`−133.85618156472032`, `−134.12866507174962`,
  `−134.25790122256726`), cycle 3+ agree to ≤3e-14. The level-shift change
  is exactly what T03 claims.
- Independent upstream probe (same recipe): `converged False, cycles 80,
  e −134.43322882555535` — upstream does not converge either.
- `YTA_MAXCYC=150`: s1 AND s2 still unconverged at 150 (s1 e
  −134.4520035716, s2 e −134.4521860784).
- T07 code changes are kept (build passes, `"final":false` history lines
  prove the hook wiring); only the verify recipe's cycle budget fails.
  T06's oracle proves the final-diagonalisation mechanism against upstream
  (`e_tot |Δ| 2.2e-13`, `mo_energy max|Δ| 2.7e-8`).

Coordination: the two sessions must not share checkpoint dirs (use unique
names) and should not edit the same files concurrently — several edits in
this tree landed from both sides (e.g. duplicated `conv_check_mode` block,
T04 error-message wording). Second session proceeds to T08+ per owner
instruction; T16 (DZVP + pre-basis, a converging setup) remains the real
pipeline-vs-upstream arbiter.

## Resolution (review, 2026-10-03)

The verify recipe in the task file was wrong, not the code: a single-basis
SZV run at 3×3×1 and 1e-7 does not converge in the port or in upstream. The
task file now uses the T16 recipe (DZVP from an SZV `pre`, 3×3×3, 1e-6).
With it, on a fresh checkpoint directory (`.yta_bundle/upcheck/t16r`):

- `exit=0`; `"final":true` appears once in `history.jsonl`; the log has one
  `periodic SCF final diagonalisation` line (stage `s2` only).
- A second run in the same directory skips `s1` and `s2` ("converged in an
  earlier session"), reuses the 18 band points, `exit=0`, and
  `summary.gap_ev` is unchanged (2.2857852805356953 both times).

T07 is done.

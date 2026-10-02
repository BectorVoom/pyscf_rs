# T07 — Final diagonalisation: the band pipeline

**Goal.** `crates/pyscf-pbc-dft/examples/yta7o19_bands.rs` uses the final
step for the LAST SCF stage only, checkpoints it, and old checkpoints of the
other stages stay valid.

**Why last stage only.** Stages `pre` and `s1` run with a level shift and a
wide smearing. After the final step upstream itself reports them as not
converged (measured on KTaO3). They only produce a starting density.

## Read first

`crates/pyscf-pbc-dft/examples/yta7o19_bands.rs`:
- the header table of environment variables (lines ~38–68);
- `struct Stage` and `impl Stage` (search `struct Stage`, `fn identity`,
  `fn clone_controls`);
- `fn run_stage` (the `hook` closure and the `KScfConfig { … }` literal).

## Do

1. Header table: add the row

```
//! | `YTA_CONV_CHECK` | `final` | final diagonalisation after convergence (`scf/hf.py:211-232`): `final` = last SCF stage only, `all`, or `off` |
```
2. `struct Stage`: add `conv_check: bool`. In `clone_controls` copy it.
3. `fn identity`: when `self.conv_check` is `true`, add the key
   `"conv_check": true` to the JSON. When it is `false`, the JSON must be
   byte-for-byte what it is today (old checkpoints depend on that string).

```rust
let mut v = json!({"stage": self.name, "smearing": self.smearing, "conv_tol": conv_tol, "upstream": upstream});
if self.conv_check {
    v["conv_check"] = json!(true);
}
v.to_string()
```
4. In `main`, read the variable once:

```rust
let conv_check_mode: String = env_or("YTA_CONV_CHECK", "final".to_string());
assert!(matches!(conv_check_mode.as_str(), "final" | "all" | "off"), "YTA_CONV_CHECK: final|all|off");
let has_s2 = std::env::var("YTA_REFINE_SMEARING").ok().is_some_and(|v| !v.trim().is_empty());
```
   and set, where each `Stage { … }` is built:
   - `pre`: `conv_check: conv_check_mode == "all"`
   - `s1`: `conv_check: conv_check_mode == "all" || (conv_check_mode == "final" && !has_s2)`
   - `s2`: `conv_check: conv_check_mode != "off"`
5. `run_stage`, the `KScfConfig { … }` literal: add
   `conv_check: stage.conv_check,`.
6. `run_stage`, the `hook` closure: add `"final": st.final_state` to the
   JSON passed to `ck.history(…)`. Leave `save_state` as it is — the final
   call overwrites the state file with the final density and the final
   `converged` flag, which is what a later session must see.
7. `tools/kaggle-t4/upstream_check.py`, function `compare`: the trajectory
   list must skip the final line. Change
   `traj = [h["e_tot"] for h in history if h["stage"] == name]` to
   `traj = [h["e_tot"] for h in history if h["stage"] == name and not h.get("final")]`.
8. `tools/kaggle-t4/config.example.json`: add `"YTA_CONV_CHECK": "final"`
   to `env`.

## Verify

1. `cargo build --release -p pyscf-pbc-dft --example yta7o19_bands -j12`
   → `Finished`.
2. Run the pipeline on KTaO3 (about 5 minutes; this is the recipe that
   converges — a single-basis SZV run at 3×3×1 and 1e-7 does not, in the
   port or in upstream):

```bash
U=/home/user/Documents/workspace/.yta_bundle/upcheck; rm -rf $U/t07; mkdir -p $U/t07
env YTA_CELL=$U/ktao3.json YTA_CKPT_DIR=$U/t07 YTA_BASIS=gth-dzvp-molopt-sr \
    YTA_PRE_BASIS=gth-szv-molopt-sr YTA_PRE_CONV=1e-4 YTA_KE=60 \
    YTA_KMESH=3,3,3 YTA_NPATH=8 YTA_CONV=1e-6 YTA_SMEARING=gauss:0.03 YTA_DAMP=0.7 \
    YTA_LEVEL_SHIFT=0.15 YTA_DIIS_START=8 YTA_DIIS_SPACE=16 \
    YTA_REFINE_SMEARING=gauss:0.002 YTA_REFINE_LEVEL_SHIFT=0.05 \
    target/release/examples/yta7o19_bands 2> $U/t07/run.log; echo exit=$?
grep -c '"final":true' $U/t07/history.jsonl
grep -c "periodic SCF final diagonalisation" $U/t07/run.log
```
   Expected: `exit=0`; both greps print `1` (only stage `s2` does the final
   step).
3. Run the same command a second time WITHOUT deleting `$U/t07`. Expected
   in `run.log`: `stage s1: converged in an earlier session` and
   `stage s2: converged in an earlier session`, `exit=0`, and
   `result.json` unchanged in `summary.gap_ev`.

## If it fails

- Step 3 restarts a stage: its identity string changed between the two
  runs. Print both (`[yta] stage …: controls changed (a -> b)` in the log)
  and make them equal; do not delete the check.

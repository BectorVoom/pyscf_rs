# T15 — Pipeline fingerprint, configuration, documentation

**Goal.** The band pipeline records which pseudopotential route produced a
checkpoint, refuses to mix routes, and the T4 configuration carries the new
knobs.

## Read first

- `crates/pyscf-pbc-dft/examples/yta7o19_bands.rs`: search `let fingerprint`
  and read to the end of that `if let Some(fp)` block; search
  `result["method"] = json!`; search `"controls": {`.
- `tools/kaggle-t4/config.example.json`, `tools/kaggle-t4/README.md`.

## Do

1. Example, fingerprint. Change `let fingerprint = json!({…});` to
   `let mut fingerprint = json!({…});` and add directly after it:

```rust
// A checkpoint without this key was written with the analytic (real-space)
// non-local pseudopotential — its one-electron matrices differ.
let pp_reciprocal =
    pyscf_pbc_df::pp_gspace::pp_nonlocal_route() == pyscf_pbc_df::pp_gspace::PpNonlocal::Reciprocal;
if pp_reciprocal {
    fingerprint["pp_nonlocal"] = json!("reciprocal");
}
```
   Do not change the comparison code below it. Result: an old checkpoint
   (no key) is refused by a default run, and accepted when the user sets
   `PYSCF_PBC_FFTDF_PP_NL=realspace`.
2. Example, result: after the line that sets `result["method"]`, add
   `result["method"]["pp_nonlocal"] = json!(if pp_reciprocal { "reciprocal" } else { "realspace" });`
   (move the `pp_reciprocal` definition above it if needed).
3. Example, stage report: in the `"controls": {…}` JSON of `run_stage` add
   `"conv_check": cfg.conv_check`.
4. Example header table, add two rows:

```
//! | `PYSCF_PBC_FFTDF_PP_NL` | unset | `realspace`: the analytic non-local pseudopotential (the route of runs made before 2026-10); default is upstream's reciprocal-space route |
//! | `PYSCF_PBC_PP_NL_BUDGET_MB` | `1024` | device memory for one G-block of the reciprocal-space route |
```
5. Example, `System::from_env`: let `YTA_CELL` also carry the JSON itself
   (a Kaggle notebook has no place to put an extra file). Where the file is
   read, use:

```rust
let raw = if path.trim_start().starts_with('{') {
    path.clone()
} else {
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("YTA_CELL {path}: {e}"))
};
```
   and update the `YTA_CELL` row of the header table: "a JSON file, or the
   JSON text itself".
6. `tools/kaggle-t4/upstream_check.py`, function `make_mf`: replace
   `mf.conv_check = not like` by
   `mf.conv_check = False if like else controls.get("conv_check", True)`.
   In `main`, change the default of `--arms` from `"like,stock"` to
   `"stock"`. Add to the module docstring, as its last paragraph:
   `Results written by a runner with the three fixes (method.pp_nonlocal ==
   "reciprocal") need only the stock arm.`
7. `tools/kaggle-t4/config.example.json`, in `env`:
   - `"YTA_LEVEL_SHIFT": "0.15"` (was `0.3`) and
     `"YTA_REFINE_LEVEL_SHIFT": "0.05"` (was `0.1`);
   - add `"PYSCF_PBC_PP_NL_BUDGET_MB": "1024"`.
8. `tools/kaggle-t4/README.md`: add this section before
   "Checking a run against upstream PySCF":

```markdown
## Changes that affect old runs (2026-10)

- **Level shift.** The shift now follows PySCF: for a restricted run it
  lowers occupied levels and raises virtual levels by the shift. To keep
  the convergence an older run had, use HALF the old value (0.3 → 0.15).
- **Non-local pseudopotential.** Default is PySCF's reciprocal-space
  route. A checkpoint from an older runner is refused; either start a new
  run or set `PYSCF_PBC_FFTDF_PP_NL=realspace` in `env` to continue it.
- **Final diagonalisation.** The last SCF stage re-diagonalises the plain
  Fock matrix after convergence (`YTA_CONV_CHECK`), as PySCF does.
```

## Verify

1. `cargo build --release -p pyscf-pbc-dft --example yta7o19_bands -j12`.
2. `python3 -W ignore -m unittest discover -s tools/kaggle-t4/tests` → `OK`.
3. Old checkpoint is refused, then accepted with the opt-in:

```bash
U=/home/user/Documents/workspace/.yta_bundle/upcheck; rm -rf $U/t15; cp -r $U/ktao3_cpu $U/t15
E="YTA_CELL=$U/ktao3.json YTA_CKPT_DIR=$U/t15 YTA_BASIS=gth-dzvp-molopt-sr YTA_PRE_BASIS=gth-szv-molopt-sr YTA_KE=60 YTA_KMESH=3,3,3 YTA_STOP_AFTER=1e"
env $E target/release/examples/yta7o19_bands > /dev/null 2>&1; echo "default exit=$?"
env $E PYSCF_PBC_FFTDF_PP_NL=realspace target/release/examples/yta7o19_bands > /dev/null 2>&1; echo "realspace exit=$?"
```
   Expected: `default exit=6`, `realspace exit=0`.

## If it fails

- `default exit=0`: the key was not added to the fingerprint, or the
  comparison ignores unknown keys — read the comparison code and report
  what it compares; do not rewrite it.

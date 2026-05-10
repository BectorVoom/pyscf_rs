# pyscf-rs Environment Variables

Single source of truth for the env-var surface pyscf-rs honours. Every variable
listed here is paired with a runtime resolver in the corresponding crate; tests
in that crate verify the resolver under `default`, `set`, `unset`, and
`malformed` regimes.

| Var                | Phase   | Default                              | Purpose                                                                                                                                                                                                                       |
| ------------------ | ------- | ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `PYSCF_BACKEND`    | 1 (D-07) | `auto`                              | Selects compute backend. Values: `cpu`, `cuda`, `wgpu`, `rocm`, `metal`, `auto`. Unrecognised falls back to CPU with a `tracing::warn!`. Resolver: `pyscf-algebra::select`.                                                  |
| `PYSCF_DTYPE`      | 1 (D-08) | `f64`                               | Selects kernel precision. Values: `f32`, `f64` (case-insensitive). With `PYSCF_BACKEND=wgpu` and `f64` on an adapter without `shader-f64`, the resolver hard-errors per Phase 1 D-09.                                          |
| `PYSCF_BASIS_PATH` | 2 (D-02) | walk-up to `pyscf/gto/basis/`       | Overrides the built-in basis-set file directory. Set this when running pyscf-rs from a maturin wheel where the upstream `pyscf/gto/basis/` directory is bundled at a non-default location. Resolver: `pyscf-gto::basis::path`. |

## Resolution priorities

### `PYSCF_BACKEND` (Phase 1 D-07)

1. If `PYSCF_BACKEND` is set and matches a known backend with a probe-passing
   adapter → use it.
2. If `PYSCF_BACKEND=auto` → probe in order: cuda → rocm → wgpu/metal → cpu;
   first that succeeds wins.
3. Unknown / unset / probe-fail → CPU, with `tracing::warn!` if the user
   explicitly requested a non-CPU backend.

### `PYSCF_DTYPE` (Phase 1 D-08)

1. Honour `PYSCF_DTYPE` if set to `f32` / `f64` (case-insensitive).
2. Default `f64`.
3. With `PYSCF_BACKEND=wgpu` + `PYSCF_DTYPE=f64`, resolver checks the wgpu
   adapter for `shader-f64`. Missing → hard-error per D-09 (do NOT silently
   downgrade to f32 — it would lie to the chemistry user).

### `PYSCF_BASIS_PATH` (Phase 2 D-02)

1. Honour `PYSCF_BASIS_PATH` if set and the path is a directory.
2. Walk up from `CARGO_MANIFEST_DIR` looking for `../../pyscf/gto/basis/`
   (dev mode).
3. Walk up from `current_exe()` looking for `../pyscf/gto/basis/` (installed
   wheel layout).
4. Error with a `BasisLoadError::PathNotFound` naming `PYSCF_BASIS_PATH`.

## Test setup

The Phase 2 oracle harness at `tests/oracle/` requires upstream PySCF +
numpy + pytest:

```bash
pip install -r tests/oracle/requirements.txt
```

If the dev box's Python lacks SSL (e.g., a stripped pyenv build with
`ModuleNotFoundError: No module named 'ssl'`), install via the system package
manager instead:

```bash
# Debian / Ubuntu
sudo apt install python3-numpy python3-pytest python3-scipy
# Then pip-install pyscf into a venv that uses the system Python
python3 -m venv --system-site-packages .venv
.venv/bin/pip install pyscf
```

Or download the pyscf wheel via a different path and `pip install ./pyscf-2.6-*.whl`.

`tests/oracle/conftest.py` collects-as-skipped if `import pyscf` fails, so the
absence of these prereqs only suppresses the byte-identity tests rather than
breaking the wider test run.

When launching the oracle suite, set `PYSCF_BASIS_PATH` to the vendored
basis directory:

```bash
export PYSCF_BASIS_PATH="$(pwd)/pyscf/gto/basis"
pytest tests/oracle/
```

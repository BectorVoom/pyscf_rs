# Contributing to pyscf-rs

Welcome. pyscf-rs is the pure-Rust rewrite of PySCF; this file is the day-one
orientation for new contributors and the reference for the local
sibling-crate development workflow.

## Project layout

The Rust workspace lives at the repo root and coexists with the upstream
PySCF Python tree (`pyscf/`, `pyproject.toml`, `examples/`, `pytest.ini`)
without disturbing it. Two trees, two build systems:

| Tree              | Build      | Files                                                    |
|-------------------|------------|----------------------------------------------------------|
| Rust workspace    | `cargo`    | `Cargo.toml`, `crates/`, `xtask/`, `.cargo/config.toml`  |
| Upstream PySCF    | `pip`/`maturin` | `pyscf/`, `pyproject.toml`, `setup.py`, `pytest.ini`, `examples/` |

The Rust workspace has 15 member crates plus `xtask`:

```text
crates/pyscf-rs        # top-level façade — re-exports core/runtime/algebra
crates/pyscf-core      # universal types and method traits (no compute deps)
crates/pyscf-runtime   # BackendKind, per-backend probes, WorkspacePool, tracing
crates/pyscf-algebra   # SOLE cubecl-* consumer (alongside pyscf-runtime per ALG-06)
crates/pyscf-{kernels,gto,scf,dft,mp2,ccsd,grad,geomopt}  # method crates (Phases 2-7)
crates/pyscf-py        # PyO3 abi3-py310 wheel surface (Phase 3)
crates/pyscf-oracle    # PySCF live oracle in dev-deps (Phase 3)
crates/pyscf-bench     # criterion benchmark suite (Phase 8)
xtask/                 # internal lint binaries (NOT part of the 15-member tally)
```

## Local sibling-crate development (D-15)

pyscf-rs's `[patch.crates-io]` block points `cintx`, `libxc_rs`, and
`xcfun_rs` at GitHub HEAD. For day-to-day development against local
sibling-crate checkouts, use a developer-local `~/.cargo/config.toml`
override (NOT shipped in the repo):

```toml
# ~/.cargo/config.toml — replace <user> with your username and confirm
# the paths exist locally.
[patch.crates-io]
cintx     = { path = "/home/<user>/Documents/workspace/cintx" }
libxc_rs  = { path = "/home/<user>/Documents/workspace/libxc_rs" }
xcfun_rs  = { path = "/home/<user>/Documents/workspace/xcfun_rs" }
```

With this in place, `cargo build` resolves cintx/libxc_rs/xcfun_rs from
your local clones; the workspace's `[patch.crates-io]` Git pins are
overridden by your user-level config. To switch back to the pinned Git
revs, comment out the user-level `[patch.crates-io]` block.

## CI gates (Plan 05 + Plan 06)

Five xtask binaries gate every PR merge. Run any of them locally before
pushing:

| Binary                       | What it checks                                                              | Maps to            |
|------------------------------|-----------------------------------------------------------------------------|--------------------|
| `check-no-fma`               | `release-oracle` machine code has zero FMA mnemonics                        | FOUND-05, Pitfall 1 |
| `check-forbidden-paths`      | No imports from upstream PySCF out-of-scope modules (pbc/x2c/mcscf/...)     | FOUND-08, Pitfall 21 |
| `check-catch-unwind`         | Every `extern "C"` block has `catch_unwind`                                 | FOUND-07, Pitfall 14 |
| `check-dependency-wall`      | Only `pyscf-algebra` and `pyscf-runtime` may depend on `cubecl-*`           | ALG-06              |
| `check-cubecl-pin`           | cubecl-{cpu,cuda,hip,wgpu,runtime}=0.10.0; cubecl-{matmul,reduce}=0.9.0-pre.5 | FOUND-04, Pitfall 1 |

Invoke locally:
```bash
cargo run -p xtask --bin check-no-fma
cargo run -p xtask --bin check-forbidden-paths
cargo run -p xtask --bin check-catch-unwind
cargo run -p xtask --bin check-dependency-wall
cargo run -p xtask --bin check-cubecl-pin

# Or run all five sequentially:
cargo run -p xtask
```

Plus the standard checks:
```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo test --workspace --locked -- --test-threads=1
cargo deny check
```

The oracle-determinism contract (Roadmap success criterion 3) is a
matrix job in CI: the same test runs with `RAYON_NUM_THREADS=1` and
`RAYON_NUM_THREADS=8`, both under `--profile release-oracle`, and the
bit-pattern of `oracle_sum`'s output must match. Run locally:
```bash
RAYON_NUM_THREADS=1 cargo test --profile release-oracle -p pyscf-algebra --test oracle_determinism --locked
RAYON_NUM_THREADS=8 cargo test --profile release-oracle -p pyscf-algebra --test oracle_determinism --locked
```

## Backend selection at runtime

Two env vars drive the algebra layer:

| Env var          | Default | Values                                              | Effect |
|------------------|---------|-----------------------------------------------------|--------|
| `PYSCF_BACKEND`  | `cpu`   | `cpu` / `cuda` / `wgpu` / `rocm` / `metal` / `auto` | Selects the AlgebraClient backend. `auto` walks the priority chain `cuda → rocm → metal → wgpu → cpu`. Unrecognised values fall back to CPU with a `tracing::warn!`. |
| `PYSCF_DTYPE`    | `f64`   | `f32` / `f64`                                       | Floating-point precision. `wgpu`+`f64` requires the `shader-f64` Vulkan extension; explicit `wgpu`+`f64` without it returns a hard error. |

Per ALG-08, every PyO3 entry point emits a `tracing::info!` line on
backend resolution: `pyscf-algebra: backend=cpu (env=unset, dtype=f64)`.

## Cubecl pin upgrade

cubecl 0.10.0 is exact-pinned across the workspace AND across the
sibling crates (cintx, libxc_rs, xcfun_rs). Bumping cubecl is a
four-crate operation; see [docs/upgrade-cubecl.md](docs/upgrade-cubecl.md)
for the documented ritual.

## Code style

- `rustfmt` defaults; `cargo fmt` before every commit.
- `clippy::unwrap_used` is `warn` per-crate and `deny` in CI; never use
  `.unwrap()` in numerical modules (FOUND-07).
- `#![forbid(unsafe_code)]` is enforced in `pyscf-core`; other crates
  MAY use `unsafe` only with documented invariants.
- Comments cite REQ-IDs (`FOUND-05`, `ALG-04`, etc.) and CONTEXT decision
  IDs (`D-04`, `D-13`) to make traceability greppable.

## Phase planning

Work is organised into 8 phases per `.planning/ROADMAP.md`. Each phase
has a `PHASE_N-PLAN.md` plan series in `.planning/phases/N-{name}/`.
See `.planning/CLAUDE.md`-style workflows for the complete plan/execute
flow.

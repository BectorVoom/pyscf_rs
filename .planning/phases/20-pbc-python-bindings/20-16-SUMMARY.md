# 20-16 SUMMARY — PyO3 wall (D-PBC-14) machine-checked beside the cubecl wall

**Shipped:** 2026-09-13. All four tasks complete. One file changed:
`xtask/src/bin/check_dependency_wall.rs` (+74 / −3). No CI YAML change, no
production (`crates/`) change, no commit (D-20-A).

## Task 1 — the pyo3 rule

- New `PYO3_FORBIDDEN_DEPS = ["pyo3", "pyo3-build-config", "numpy"]` and
  `PYO3_ALLOWED_CRATES = ["pyscf-py", "pyscf-oracle"]`, checked in a separate
  loop over `cargo metadata --no-deps` before the (unchanged) cubecl loop.
- Scope is **stricter than the cubecl rule on purpose**: every workspace member
  (not only `pyscf-*`), and dependencies of **every kind** (normal, dev,
  build) — a `pyo3-build-config` build-dep or a pyo3 dev-dep in a PBC crate is
  still a breach of "pyo3-free". The message names the crate, the dep kind and
  the dep.
- `FORBIDDEN_DEPS` / `ALLOWED_CRATES` and the cubecl loop are byte-identical to
  before; the only removed lines are the old exit-code doc line and the early
  `Ok(ExitCode::SUCCESS)` (now one exit decision after both reports).
- Exit contract kept: `0` pass (both rules), `1` metadata error (anyhow bail),
  `2` any violation of either rule.

### Deviation — pre-existing exception found (`pyscf-oracle`)

The plan said only `pyscf-py` names pyo3. `cargo metadata` at kickoff found a
second crate:

| crate | dep | kind | optional |
|---|---|---|---|
| `pyscf-py` | `pyo3`, `numpy` | normal | no |
| `pyscf-oracle` | `pyo3` (`features = ["auto-initialize"]`) | normal | **yes**, behind non-default `python` feature |

`pyscf-oracle` is the upstream-PySCF test oracle (in-process Python driver). It
is a dev-dependency of every crate that uses it (2 edges, both `kind = dev`)
and never a normal dependency of `pyscf-py` or a method crate (ORACLE-01). It
is allowlisted explicitly with a comment citing this, so the lint's first run
is green without weakening the rule for method crates. No `pyscf-pbc-*` crate
names pyo3/numpy.

## Task 2 — proven red on an injected dep

Snapshot `crates/pyscf-pbc-gto/Cargo.toml` (carries uncommitted Phase-18 edits;
sha256 `4abb449e…f45e`), inserted `pyo3 = { workspace = true }` as the first
line of `[dependencies]`, ran the built binary directly
(`target/debug/check-dependency-wall`, so `cargo run` could not re-resolve and
rewrite `Cargo.lock`):

```
check-dependency-wall: PASS — cubecl-* containment intact (ALG-06)
check-dependency-wall: FAIL — D-PBC-14 PyO3 wall violation:
  - pyscf-pbc-gto: declares normal dep on `pyo3` — D-PBC-14 PyO3 wall forbids; only ["pyscf-py", "pyscf-oracle"] may name ["pyo3", "pyo3-build-config", "numpy"]

Fix: keep the method crate pyo3-free; put #[pyclass]/#[pyfunction] wrappers in crates/pyscf-py.
exit=2
```

Reverted by deleting exactly that line (no git restore). `cmp` against the
snapshot → **byte-identical**; sha256 unchanged (`4abb449e…f45e`); `git diff`
still shows only the Phase-18 `default-features = false` self-dev-dep hunk.
`Cargo.lock` sha256 was unchanged across the injected run.

## Task 3 — the cubecl rule still behaves

- Direct deps: 0 of 19 `pyscf-pbc-*` crates name any `cubecl-*`/`cube-math`;
  all 19 take `pyscf-algebra` as a normal dep.
- Transitive: `cargo tree --locked --offline -p <crate> -e normal | grep -c ' cubecl v'`
  is non-zero for **all 19** (6–12 lines each) — cubecl is transitive
  everywhere, and the direct-deps wall stays green, as designed.
- Red still works and the two rules are independent: injecting
  `cubecl = { workspace = true }` into `pyscf-pbc-gto` gave
  ```
  check-dependency-wall: FAIL — ALG-06 violation:
    - pyscf-pbc-gto: declares normal dep on `cubecl` — ALG-06 forbids; only ["pyscf-algebra", "pyscf-runtime", "pyscf-kernels", "pyscf-bench"] may consume cubecl-*
  ...
  check-dependency-wall: PASS — PyO3 wall intact (D-PBC-14)
  exit=2
  ```
  then reverted the same way (byte-identical again, `cmp` clean).

## Task 4 — CI

`ci.yml:170` job `xtask-dependency-wall` runs
`cargo run --quiet -p xtask --bin check-dependency-wall` — the same binary — so
it enforces the new rule with **no YAML change**. `xtask/src/main.rs` already
lists `check-dependency-wall` in its registry.

## Verification (2026-09-13)

- Green run, clean tree (after both reverts):
  ```
  $ cargo run --quiet -p xtask --bin check-dependency-wall
  check-dependency-wall: PASS — cubecl-* containment intact (ALG-06)
  check-dependency-wall: PASS — PyO3 wall intact (D-PBC-14)
  exit=0
  ```
- Injected `pyo3` into `pyscf-pbc-gto` → exit **2**, names crate + dep; reverted.
- `cargo run -p xtask` (all checks) → exit **0**: all six registered checks OK (`check-no-fma` 6m54s
  `release-oracle` build, `check-forbidden-paths`, `check-catch-unwind`,
  `check-dependency-wall` with both PASS lines, `check-cubecl-pin`,
  `check-orphan-modules`).
- `grep -rn 'pyo3' crates/pyscf-pbc-*/Cargo.toml` → no matches (exit 1).
- `git diff --stat -- crates/` → no change from this plan (`pyscf-pbc-gto/Cargo.toml`
  byte-identical to its pre-plan working-tree state).
- `rustfmt --edition 2024 xtask/src/bin/check_dependency_wall.rs` only.
- No test file added: the lint's behaviour is pinned by the two injected red
  runs above; the xtask has no library seam for `cargo metadata` parsing.

## Notes

- `Cargo.lock` changed during this plan's window (sha `18c4e2ad…` →
  `63bcfbca…`) through concurrent Phase-20 work in the tree; its
  `pyscf-pbc-gto` entry contains no `pyo3`, and this plan's own injected runs
  were verified not to touch it.

# Cubecl upgrade ritual (FOUND-04, D-14)

cubecl 0.10.0 is exact-pinned across the pyscf-rs workspace AND across
the three sibling crates (cintx, libxc_rs, xcfun_rs). Any cubecl bump
is a **four-crate ABI contract** — the same `cubecl-runtime` major
version must be in scope across all four workspaces simultaneously,
or `cargo build --workspace` will refuse to resolve.

This document is the documented ritual referenced by FOUND-04 and the
D-14 nightly-cross-crate.yml automation.

## When does this trigger?

cubecl ships breaking changes regularly (it's pre-1.0). Triggers:

1. Tracel-AI publishes cubecl 0.11.0 (or any non-patch bump).
2. A sibling crate (cintx, libxc_rs, xcfun_rs) bumps its cubecl pin.
3. The nightly `.github/workflows/nightly-cross-crate.yml` job goes
   red because `cargo update -p cintx -p libxc_rs -p xcfun_rs` pulled
   a sibling that moved its cubecl pin without us.

## The ritual

Bumps must land in **lockstep order** to avoid intermediate
`cargo build` failures:

### Step 1 — Verify the new cubecl version is available

```bash
cargo info cubecl                      # confirm 0.11.0 published
cargo info cubecl-runtime              # confirm matching version
cargo info cubecl-{cpu,cuda,hip,wgpu}  # confirm all backends published
cargo info cubecl-matmul               # confirm matmul/reduce match (currently 0.9.0-pre.5 skew)
cargo info cubecl-reduce
```

If any backend is missing at the new version, **stop**. The four-crate
family must move together; you cannot bump pyscf-rs alone.

### Step 2 — Bump cintx first (it's the integral engine other siblings depend on)

1. Open `~/Documents/workspace/cintx`
2. Update `[workspace.dependencies]` cubecl-* pins to the new version
3. Update any `#[cube]`-using source files for breaking API changes
4. Run `cargo build --workspace` and `cargo test --workspace`
5. Commit, push, tag the new revision SHA: `git rev-parse HEAD`

### Step 3 — Bump libxc_rs and xcfun_rs in parallel

Same procedure as cintx (steps 1-5). These two crates do not depend on
each other, so they can move concurrently. Note their new revision
SHAs.

### Step 4 — Bump pyscf-rs last

1. Edit `Cargo.toml` `[workspace.dependencies]` cubecl-* pins to the new version
2. Edit `Cargo.toml` `[patch.crates-io]` to point each sibling at its
   new revision SHA from steps 2-3
3. Update `xtask/src/bin/check_cubecl_pin.rs` `REQUIRED_VERSION`
   constant to the new version (and `PRE_REQUIRED_VERSION` if
   cubecl-matmul/cubecl-reduce moved)
4. Update `crates/pyscf-algebra/src/*.rs` for any breaking cubecl API
   changes (most likely: `cubecl::client::ComputeClient<R>` path drift,
   `cubecl_matmul::Strategy` rename, etc.)
5. Run `cargo build --workspace --locked` and the test suite
6. Run `cargo run -p xtask --bin check-cubecl-pin` to confirm the new
   pin holds across the dep graph

### Step 5 — Commit and observe nightly-cross-crate

The `.github/workflows/nightly-cross-crate.yml` job runs at 06:00 UTC
and re-bumps the sibling-crate revs to current HEAD. After your
pyscf-rs PR merges, the next nightly run is the regression-check; if
it goes green, the lockstep is restored.

## What if a sibling crate breaks lockstep unilaterally?

The `nightly-cross-crate.yml` workflow's `check-cubecl-pin` step will
fail with a clear diagnostic. The recovery is:

1. **Block the merge** of any pyscf-rs PR that would compound the
   drift (e.g., a Phase 4 DFT plan that adds a new method-crate cubecl
   call site).
2. **Bisect** which sibling moved by inspecting the
   `cross-crate-cargo-lock` artifact uploaded on the failing run.
3. **Decide**: bump pyscf-rs to follow (run this ritual from step 1),
   OR pin the offending sibling at its previous revision in
   `[patch.crates-io]` (use `rev = "<prior_sha>"` instead of `branch =
   "main"`).

## Why this ritual exists

cubecl's pre-1.0 ABI moves between point releases (RESEARCH Pitfall 3).
Without this ritual, a pyscf-rs build would intermittently fail
depending on which sibling crate's cubecl version cargo's solver
happened to pick. The lockstep makes the version one workspace-wide
decision rather than per-crate negotiation.

Reference: ROADMAP success criterion 4; CONTEXT D-12, D-13, D-14;
`.github/workflows/nightly-cross-crate.yml`; `xtask/src/bin/check_cubecl_pin.rs`.

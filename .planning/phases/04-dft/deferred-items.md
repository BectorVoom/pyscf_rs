# Phase 04 — Deferred / Out-of-Scope Items

Discovered during execution but NOT in scope of the current plan's changes.
Per the executor SCOPE BOUNDARY rule, these are logged here and left unfixed.

## 04-10

- **`check-cubecl-pin` FAIL: `cubecl-hip-sys: version 7.1.5280200` flagged as
  "cubecl-* family member outside transitive carve-out"** (FOUND-04 / Pitfall 1
  lint, `xtask/src/bin/check_cubecl_pin.rs`).
  - **Pre-existing:** present in `Cargo.lock` at the 04-10 Task-1 commit
    (`cubecl-hip-sys` count > 0 before the libxc-patch re-enable). NOT caused by
    re-enabling the `[patch.crates-io] libxc_rs` entry — the libxc patch is
    inert in the default build (`cargo tree` shows 0 libxc_rs) and the lockfile
    is unchanged by it.
  - **Nature:** `cubecl-hip-sys` is a transitive `*-sys` system-binding crate
    pulled by `cubecl-hip`; its version string (`7.1.5280200`, a HIP toolkit
    version) is not one of the workspace cubecl pins, so the lint's
    name-prefix match treats it as drift. Likely a lint false-positive (the
    `-sys` crate is not part of the cubecl 0.10.0 lockstep set) rather than a
    real cubecl version-skew regression.
  - **Disposition:** Out of scope for 04-10 (touches the libxc/wgpu CI surface +
    the WGPU f64 honesty path, not the cubecl-pin lint or the HIP sys dep).
    Should be triaged separately — either (a) add `cubecl-hip-sys` to the lint's
    transitive carve-out / `*-sys` exemption, or (b) align the pin if it is a
    genuine skew. Do NOT trigger any libxc build while investigating.

## 04-11

- **`pyscf-dft` `tests/wgpu_f64_fallback.rs::wgpu_f64_fallback` is flaky** — failed
  once during the 04-11 Task-2 batch run (`cargo test -p pyscf-kernels -p pyscf-gto
  -p pyscf-dft`), then passed deterministically on an isolated re-run (both subtests
  green).
  - **Pre-existing & unrelated:** 04-11 modifies only `crates/pyscf-kernels/src/eval_gto.rs`,
    `crates/pyscf-gto/src/eval_gto.rs`, and `crates/pyscf-kernels/tests/eval_gto_lge1.rs`.
    `git diff --name-only` confirms `pyscf-dft/` is untouched by this plan.
  - **Nature:** WGPU-backend probing + `PYSCF_BACKEND` env-var sensitivity. Matches the
    project-memory note on flaky env-var tests (serialized in commit fb17f77); the flake
    surfaces under cross-crate parallel test scheduling, not from any code path this plan
    changes.
  - **Disposition:** Out of scope for 04-11 (CR-03 l>4 panic→Result). Triage the WGPU
    backend-probe / env-var test isolation separately.

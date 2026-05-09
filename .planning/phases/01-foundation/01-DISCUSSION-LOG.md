# Phase 1: Foundation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in 01-CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-10
**Phase:** 1-foundation
**Areas discussed:** Workspace location & repo restructure, AlgebraClient dispatch shape, `auto` backend resolution policy, Sibling-crate sourcing for [patch.crates-io]

---

## Workspace location & repo restructure

| Option | Description | Selected |
|--------|-------------|----------|
| Root coexistence (cintx pattern) | Cargo.toml + crates/ + xtask/ at the root, sitting alongside the existing upstream pyscf/ Python tree, pyproject.toml, examples/, etc. Matches cintx exactly. Maturin can later build a wheel from the same root. | ✓ |
| Nested `rust/` subdir | Entire workspace under rust/. Keeps Rust and Python totally separate. Maturin needs path config; CI workflows split by directory. | |
| Move upstream PySCF to subdir | Move the existing pyscf/ source tree to pyscf-upstream/ (oracle reference); Rust workspace owns the root. Cleanest from Rust POV but a heavy git mv that disrupts the existing Python tree's identity. | |

**User's choice:** Root coexistence (cintx pattern).
**Notes:** Existing `pyscf/`, `pyproject.toml`, `setup.py`, `pytest.ini` stay untouched in Phase 1. They remain the upstream-PySCF oracle reference and the future maturin wheel host.

---

## AlgebraClient dispatch shape

| Option | Description | Selected |
|--------|-------------|----------|
| Enum + match dispatch (sibling pattern) | AlgebraClient enum with cfg-gated arms per compiled backend; free fns match-dispatch internally. Method crates stay non-generic. Matches cintx-cubecl/xcfun-gpu. | ✓ |
| Trait object (Box<dyn AlgebraBackend>) | Method crates hold `&dyn AlgebraBackend`. Backend selection fully dynamic; no cfg explosion in algebra. Cost: v-table call per primitive; cubecl per-runtime types may resist trait-object'ing. | |
| Generic over Runtime | algebra::gemm<R: Runtime>(...) generic; every method crate becomes generic on R. Maximum monomorphization speed. Cost: code bloat, every method-crate signature carries <R>, PyO3 boundary needs concrete R anyway. | |

**User's choice:** Enum + match dispatch.

### Follow-up: Tensor handle boundary

| Option | Description | Selected |
|--------|-------------|----------|
| Opaque BufferId owned by algebra | pyscf-algebra owns all device buffers; returns BufferId/Tensor newtype. Method crates pass references; algebra reconstructs TensorHandle<R> internally. Method crates never touch cubecl types. | ✓ |
| Per-backend TensorH enum | Wrapper enum mirroring AlgebraClient: TensorH::Cpu(TensorHandle<CpuRuntime, f64>), etc. Match-arm boilerplate per primitive (must match-arm assert variant agreement). | |
| Raw bytes + shape boundary | Algebra accepts &[u8] + shape + dtype on every call; reconstructs TensorHandle inside, copies result back. Every call pays alloc+copy cost — wrong for hot SCF/CCSD loops. | |

**User's choice:** Opaque BufferId + Tensor newtype owned by algebra.

---

## `auto` backend resolution policy

| Option | Description | Selected |
|--------|-------------|----------|
| Probe + skip with tracing::info per attempt | Walk priority chain compile-feature-AND-device-available; emit tracing::info per skipped backend. Loud-by-default + observable. | ✓ |
| Feature-only — first compiled wins, fail at first launch | auto picks first feature-compiled backend without probing. Failure mode is kernel-launch error, not silent CPU fallback. | |
| Probe silently — only log final selection | Same probe-and-skip, but skipped backends don't log. Quieter; runs counter to FOUND-09 verbosity contract. | |

**User's choice:** Probe + skip with tracing::info per attempt.

### Follow-up: wgpu + f64 + missing shader-f64 rule

The user clarified that f32/f64 is a separate user-controlled axis, not just an automatic backend choice. Original "where does shader-f64 check live" question was reframed.

| Option | Description | Selected |
|--------|-------------|----------|
| Explicit=hard stop, auto=skip-with-info | PYSCF_BACKEND=wgpu (explicit) + f64 + missing shader-f64 → return Err / RuntimeError; refuse to silently downgrade. PYSCF_BACKEND=auto + f64 + missing shader-f64 → skip wgpu in priority chain with tracing::info. | ✓ |
| Hard stop in both modes | Even auto refuses to silently skip wgpu when shader-f64 is missing and f64 is requested — exits forcing user to set PYSCF_DTYPE=f32 or PYSCF_BACKEND=cpu. | |

**User's choice:** Explicit=hard stop, auto=skip-with-info.
**Notes:** This decision introduces `PYSCF_DTYPE` ∈ {f32, f64} as a new env var in Phase 1, sibling to `PYSCF_BACKEND`. Both env vars appear in user-facing log lines and error messages. The original framing of the shader-f64 question (Phase 1 vs Phase 4 jurisdiction) was rejected — the user's vision is that **precision** is the lever, and the error story splits cleanly by selection mode.

---

## Sibling-crate sourcing for [patch.crates-io]

| Option | Description | Selected |
|--------|-------------|----------|
| Pinned Git commit SHAs | [patch.crates-io] sets each sibling crate to `{ git = "...", rev = "<sha>" }`. CI-portable from day one; nightly matrix CI bumps SHAs and rebuilds. | ✓ |
| Local path deps for dev + git rev for CI | Two-tier: dev-local override via ~/.cargo/config.toml; project Cargo.toml ships pinned git revs for CI. | |
| Path deps only — publish later | Workspace uses path deps to ~/Documents/workspace/*. Only works on dev machine; CI must mirror layout. Blocks ORACLE-05 portability. | |
| Crates.io publish first | Block Phase 1 until cintx/libxc_rs/xcfun_rs are published. pyscf_rs depends on registry versions; no [patch.crates-io] needed. | |

**User's choice:** Pinned Git commit SHAs.

### Follow-up: Git remote location

| Option | Description | Selected |
|--------|-------------|----------|
| GitHub under BectorVoom | git@github.com:BectorVoom/{cintx,libxc_rs,xcfun_rs}.git — same owner as this pyscf_rs repo. If remotes don't exist yet, pushing them is a Phase 1 prereq. | ✓ |
| Different owner / org | Use a different GitHub org or user (Other). | |
| No public remote yet — leave as TBD | CONTEXT.md documents intent; planner leaves URLs as TODO; Phase 1 first task is to publish each sibling repo. | |

**User's choice:** GitHub under BectorVoom.
**Notes:** Researcher / planner should verify each `https://github.com/BectorVoom/<sibling>` remote exists before locking SHAs. If a remote is missing, surface a Phase 1 prereq task to push it. The dev-local path-dep override is documented in CONTRIBUTING.md as the recommended local iteration recipe (D-15).

---

## Claude's Discretion

User did not pick options for these — the planner / researcher chooses:

- **`oracle_sum`/`oracle_dot` algorithm** — pairwise tree reduction vs Kahan-Babuska vs strict left-to-right. Constraint: bit-identical across `RAYON_NUM_THREADS=1` and `RAYON_NUM_THREADS=8` (Phase 1 success criterion 3).
- **Lint mechanism** for forbidden-paths (FOUND-08) and algebra-dependency-wall (ALG-06): dylint plugin vs xtask grep + cargo metadata jq vs cargo-deny.
- **Stub crate skeleton** for the 12 method/façade crates that aren't non-stub in Phase 1: empty lib.rs with TODO comment vs trait re-exports vs intermediate.
- **`WorkspacePool` shape** (FOUND-03): tensor buffer pool vs thread pool wrapper vs PYSCF_MAX_MEMORY-budgeted scratchpad arena vs combination. Phase 1 only needs minimal skeleton; full design in Phase 6 (CCSD-11).
- **`panic = "abort"` scope** (FOUND-07): `[profile.release]` only, or also `[profile.release-oracle]`. Default: both.

## Deferred Ideas

Mentioned during discussion but belong in later phases:

- `python/pyscf/__init__.py` re-export shim → Phase 3 (BIND-02).
- Maturin wheel build → Phase 8 (DIST-02).
- abi3-py310 wheel skeleton → Phase 3 (BIND-01).
- Full tensor-arena spill-to-HDF5 → Phase 6 (CCSD-11).
- `mol.verbose` ↔ tracing-subscriber wiring at PyO3 boundary → Phase 3.
- Per-backend GPU regression suite → Phase 8 (ORACLE-07).
- DFT kernel-side shader-f64 fallback robustness → Phase 4 (DFT-11).

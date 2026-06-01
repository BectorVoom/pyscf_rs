# Architecture Research — pyscf_rs

**Domain:** Quantum-chemistry compute library (Rust core + PyO3 bindings, cubecl on hot paths)
**Researched:** 2026-05-09
**Confidence:** HIGH (direct inspection of three sibling crates' `Cargo.toml` files; PySCF reference module map confirmed from `.planning/codebase/`)

---

## 1. Sibling-Crate Pattern Audit (observed, not assumed)

I inspected every workspace member of `cintx`, `xcfun_rs`, and `libxc_rs` and recorded the dep-graph between members. Two patterns are in use; only one is appropriate to copy.

### 1.1 cintx — 8-crate **horizontal layered façade** (the pattern to mirror)

`/home/user/Documents/workspace/cintx/Cargo.toml` declares the workspace:

- Top-level façade package `cintx` (also a workspace) — `crates/cintx-rs` and `crates/cintx-capi` are optional path-deps gated behind feature flags. The package's role is "user-visible name on crates.io, gated re-export of the safe Rust facade and the C ABI."
- `cintx-core` — types only (`Atom`, `Shell`, `BasisSet`, `EnvParams`, `Representation`, `OperatorId`, `TensorShape`, `cintxRsError`). Deps: only `thiserror`, `smallvec`. Zero compute, zero cubecl. (`crates/cintx-core/Cargo.toml` lines 9-11.)
- `cintx-ops` — operator-table / manifest resolution. Deps: `cintx-core` + serde build script. (`crates/cintx-ops/Cargo.toml` line 10.)
- `cintx-runtime` — backend-agnostic: validator, planner, scheduler, workspace allocator, dispatch table. Has feature flags for each backend (`cpu`/`wgpu`/`cuda`/`rocm`/`metal`) but pulls **no cubecl crates itself** — flags only enable typed enum arms. Deps: `cintx-core`, `cintx-ops`. (`crates/cintx-runtime/Cargo.toml` lines 17-28.)
- `cintx-cubecl` — owns every `#[cube]` kernel body. Backend feature flags here are real (pull `cubecl-wgpu`, `cubecl-cuda`, `cubecl-hip` as `optional = true` deps). Deps: `cintx-core`, `cintx-ops`, `cintx-runtime`, plus cubecl. (`crates/cintx-cubecl/Cargo.toml` lines 34-46.)
- `cintx-compat` — bridge layer between the cubecl runtime and the capi/oracle/rs crates. Forwards every backend feature. Deps: `cintx-core`, `cintx-ops`, `cintx-runtime`, `cintx-cubecl`. (`crates/cintx-compat/Cargo.toml` lines 22-27.)
- `cintx-rs` — the safe Rust facade (`SessionRequest`, `SessionBuilder`, `IntegralTensor`). Deps: `cintx-core`, `cintx-ops`, `cintx-runtime`, `cintx-cubecl`, `cintx-compat`. This is what end-users `use`. (`crates/cintx-rs/Cargo.toml` lines 10-15.)
- `cintx-capi` — `cdylib` + `rlib` C ABI shim. Deps: `cintx-core`, `cintx-compat`. (`crates/cintx-capi/Cargo.toml` lines 10-13.)
- `cintx-oracle` — vendored libcint FFI for parity testing. Deps: `cintx-core`, `cintx-ops`, `cintx-compat`, plus `bindgen`, `cc`. (`crates/cintx-oracle/Cargo.toml` lines 23-31.)

**Resulting DAG (lower depends on upper):**

```
cintx-core
   ├──► cintx-ops
   │       └──► cintx-runtime
   │               └──► cintx-cubecl
   │                       └──► cintx-compat
   │                               ├──► cintx-rs   (safe Rust facade)
   │                               ├──► cintx-capi (C ABI)
   │                               └──► cintx-oracle (parity tests)
   └──► (every crate)
```

This is a **horizontal layered façade**: dependency order is `core → ops → runtime → cubecl → compat → {rs, capi, oracle}`. There is no per-feature crate split — cintx is a single compute surface (integrals).

### 1.2 xcfun_rs — same 8-crate horizontal pattern with one rename

`/home/user/Documents/workspace/xcfun_rs/Cargo.toml` lists members `xcfun-{core, ad, kernels, eval, gpu, rs, capi, py}` plus `xtask`/`validation`. The shape is identical to cintx with two named-differences:
- `xcfun-kernels` plays the role of `cintx-cubecl` (per-functional `#[cube]` bodies). (`crates/xcfun-kernels/Cargo.toml`.)
- `xcfun-gpu` plays the role of `cintx-compat` (GPU batch lifecycle, `auto_backend` dispatch). (`crates/xcfun-gpu/Cargo.toml`.)
- `xcfun-eval` is the runtime substrate (cubecl-cpu launcher) — analogous to `cintx-runtime`.
- `xcfun-py` is the **PyO3 binding crate** that doesn't exist as a member in cintx. `crate-type = ["cdylib"]`, depends only on `xcfun-rs` + `xcfun-core` + `pyo3 = "0.28.3"` + `numpy = "0.28.0"`, and forwards every backend feature (`cpu/hip/cuda/wgpu/metal`) through to `xcfun-rs`. (`crates/xcfun-py/Cargo.toml` lines 12-26.)

**The xcfun-py wiring is the canonical PyO3 layer for our family:** one cdylib crate, default `cpu`, all GPU backends opt-in, never depends on `cubecl-*` directly.

**Resulting DAG:**

```
xcfun-core
   ├──► xcfun-ad
   │       └──► xcfun-kernels
   │               └──► xcfun-eval (cubecl-cpu substrate, default-on)
   │                       └──► xcfun-gpu (HIP/CUDA/WGPU optional deps)
   │                               ├──► xcfun-rs   (safe facade; default cpu)
   │                               │       ├──► xcfun-capi (C ABI shim)
   │                               │       └──► xcfun-py   (PyO3 cdylib)
   │                               └──► (cintx-compat counterpart)
   └──► (every crate)
```

### 1.3 libxc_rs — different pattern, do **not** copy

`/home/user/Documents/workspace/libxc_rs/Cargo.toml` is a *single top-level package* with **27 path-dep "kernel family" sub-crates** at `crates/kernels/{lda,gga,mgga,...}` plus `xtask`, `verify`, `libxc-sys`. There is no `core/runtime/cubecl/compat` split — the top-level crate owns all of it. This is appropriate for libxc_rs (one giant catalog of XC functionals) but **does not scale for pyscf_rs** because we have *multiple distinct chemistry methods* (gto, scf, dft, mp2, ccsd, grad, geomopt) each with its own non-trivial state and lifecycle.

**Verdict:** mirror the **cintx/xcfun_rs 8-crate horizontal-layered façade pattern**, not libxc_rs's flat-kernels pattern.

### 1.4 Tech-debt observations from the sibling Cargo.toml files

These inform what *not* to repeat:

| Smell | Where | What we'll do |
|---|---|---|
| Two crates conflate "compute substrate" and "validation harness" via a `testing` feature (e.g. `xcfun-gpu` requires `xcfun-eval/testing` to reach cubecl-cpu) | xcfun_rs `crates/xcfun-gpu/Cargo.toml:15-16` | Keep substrate-vs-test cleanly separate: the kernels crate has zero `dev-dependencies` on the runtime crate. |
| Top-level façade package mixes its own deps with workspace declaration (cintx sets `cintx-rs`/`cintx-capi` as optional path-deps + features `with-f12`/`with-4c1e` on the top-level package) | cintx `Cargo.toml:7-37` | Keep top-level `pyscf-rs` package thin: re-export only. Feature plumbing lives in workspace member crates. |
| Pinned versions (`= "0.10.0"`, `= "1.92"`) using *exact* equality across many deps | xcfun_rs `Cargo.toml:43-69` | Pin cubecl + pyo3 + numpy exactly (lockstep upgrades), keep utility deps loose-pinned. |
| `default-members` listing every crate inflates default `cargo build` cost | cintx `Cargo.toml:68-77` | `default-members = []` on the workspace; users pick crates. |

---

## 2. Standard Architecture for pyscf_rs

### 2.1 Two-axis crate decomposition

cintx has **one** compute surface (integrals) and uses **one** horizontal stack. pyscf_rs has **seven** chemistry methods (gto, scf, dft, mp2, ccsd, grad, geomopt) and the cintx pattern doesn't directly cover the "more than one chemistry domain" axis. The natural extension keeps the **horizontal layering per method** plus a small set of **cross-cutting infrastructure crates** at the bottom:

```
                                                                                  
  ┌────────────────────────────────────────────────────────────────────────────┐  
  │                          pyscf-py (PyO3 cdylib)                            │  
  │              one wheel; reproduces  from pyscf import gto, scf, dft        │  
  └────────────┬────────────────────────────┬────────────────────────────┬─────┘  
               │                            │                            │
  ┌────────────▼─────────────┐  ┌──────────▼──────────────┐  ┌──────────▼──────┐
  │  pyscf-rs (façade crate) │  │ pyscf-cli (binary opt)  │  │ pyscf-oracle    │
  │  re-exports method APIs  │  │ pyscf-bench (criterion) │  │ (pyo3 to upstream)│
  └────────────┬─────────────┘  └─────────────────────────┘  └─────────────────┘
               │
  ┌────────────▼──────────────────────────────────────────────────────────────┐
  │  Per-chemistry-method crates (one safe-facade crate per method)           │
  │  pyscf-gto, pyscf-scf, pyscf-dft, pyscf-mp2, pyscf-ccsd, pyscf-grad,      │
  │  pyscf-geomopt                                                            │
  └────────────┬──────────────────────────────────────────────────────────────┘
               │
  ┌────────────▼──────────────────────────────────────────────────────────────┐
  │  pyscf-kernels  (all #[cube] kernels: vhf, ao2mo, dft-numint, cc-tensor)  │
  │  Owns every cubecl kernel body. No method-state, no driver code.          │
  └────────────┬──────────────────────────────────────────────────────────────┘
               │
  ┌────────────▼──────────────────────────────────────────────────────────────┐
  │  pyscf-runtime  (planner, scheduler, workspace, backend dispatch)         │
  │  Mirror of cintx-runtime. Backend-feature gates only enable enum arms.    │
  └────────────┬──────────────────────────────────────────────────────────────┘
               │
  ┌────────────▼──────────────────────────────────────────────────────────────┐
  │  pyscf-core   (Mole, AOIntegrals, MOIntegrals, Density, MOCoefficients,   │
  │                Amplitudes, Energy, errors). Pure types and traits.        │
  └────────────────────────────────────────────────────────────────────────────┘

  External deps (path-deps from the parent workspace):
    cintx (cintx-rs)   ←  used by pyscf-gto for AO integrals
    libxc_rs           ←  used by pyscf-dft for XC functionals (LDA/GGA/mGGA)
    xcfun_rs (xcfun-rs)←  used by pyscf-dft for XC derivatives (analytic)
```

**Core insight:** the cintx "core → runtime → cubecl → compat → rs" layering applies *once*, not once per method. Each chemistry method (`pyscf-scf`, `pyscf-mp2`, …) is a thin safe-facade *on top* of the shared layering. This avoids 7×8 = 56 crates and instead gives 7 + 7 = ~14 crates total.

### 2.2 Why one `pyscf-kernels` crate, not seven `pyscf-{method}-cubecl` crates

The literal interpretation of the brief — `crates/pyscf-{module}-{rs,cubecl,oracle,py,runtime,compat}` — would yield 7 × 6 = 42 crates. That's wrong:

1. **cubecl kernels are not partitioned by chemistry method.** vhf is shared between scf and dft. ao2mo is shared between mp2, ccsd, and grad. tensor contractions are shared between mp2 and ccsd. Splitting kernels by method would force mp2 and ccsd to either (a) duplicate kernels or (b) cross-depend on each other's `*-cubecl` crate, breaking the DAG.
2. **Compile-time cost.** Each cubecl crate pulls all of `cubecl + cubecl-wgpu + cubecl-cuda + cubecl-hip` (when features are on). xcfun_rs's CLAUDE.md notes "CubeCL proc macro expansion generates massive IR" (libxc_rs profile.dev sets `codegen-units = 16` and `debug = 0` to compensate). Multiplying by 7 makes every clean build 5–10× slower.
3. **Runtime dispatch is global.** `BackendKind` and the device handle pool are program-wide, not method-specific. Putting them in one crate lets us cache one `OnceLock<CudaClient>` instead of seven.
4. **xcfun_rs already proved one kernels crate works for a heterogeneous catalog** (it ships ~40 functionals in one `xcfun-kernels` crate). Same pattern works here.

**Trade-off:** `pyscf-kernels` becomes the largest crate. Mitigate by making each method's kernels live in its own *module* (`pyscf-kernels/src/{vhf,ao2mo,numint,cc_tensor,grad}/mod.rs`), and by gating expensive kernels behind `#[cfg(feature = "with-ccsd")]`-style features so a `pyscf-scf`-only build doesn't compile the CCSD tensor kernels.

### 2.3 The seven method crates

| Crate | Owns | Depends on |
|---|---|---|
| `pyscf-gto` | `Mole`, basis-set loaders, ECP, AO integral request façade. Bridges to `cintx`. | `pyscf-core`, `pyscf-runtime`, `pyscf-kernels` (for `eval_gto` on grids), **`cintx`** (path dep). |
| `pyscf-scf` | RHF, UHF, GHF drivers, DIIS, initial guess (Hückel/MINAO/atom), Fock build orchestration. | `pyscf-core`, `pyscf-runtime`, `pyscf-kernels`, `pyscf-gto`. |
| `pyscf-dft` | RKS, UKS, Becke grid generation, Lebedev points, numerical integration, XC dispatch. | `pyscf-core`, `pyscf-runtime`, `pyscf-kernels`, `pyscf-gto`, `pyscf-scf`, **`libxc_rs`**, **`xcfun_rs`** (path deps). |
| `pyscf-mp2` | RMP2, UMP2, integral transformation (AO→MO), density-fitted MP2. | `pyscf-core`, `pyscf-runtime`, `pyscf-kernels`, `pyscf-gto`, `pyscf-scf`. |
| `pyscf-ccsd` | RCCSD, UCCSD, T1/T2 amplitude solver, DIIS for amplitudes. May import MP2 amplitudes for warm start. | `pyscf-core`, `pyscf-runtime`, `pyscf-kernels`, `pyscf-gto`, `pyscf-scf`, `pyscf-mp2`. |
| `pyscf-grad` | Analytical gradients for HF, DFT (Pople CPHF or analytic), MP2, CCSD (Λ-equation solver). | `pyscf-core`, `pyscf-runtime`, `pyscf-kernels`, `pyscf-gto`, `pyscf-scf`, `pyscf-dft`, `pyscf-mp2`, `pyscf-ccsd`. |
| `pyscf-geomopt` | Optimizer driver (BFGS via `argmin`), step control, convergence checks. Ingests forces from `pyscf-grad`. | `pyscf-core`, `pyscf-grad`. |

The DAG is acyclic: methods only depend on lower-energy methods (DFT depends on SCF; MP2/CCSD depend on SCF; grad depends on every method it differentiates; geomopt sits at the top).

### 2.4 Final crate inventory (15 workspace members)

```
pyscf_rs/
├── Cargo.toml                 # workspace root + thin `pyscf-rs` facade package
├── crates/
│   ├── pyscf-core/            # types, traits, errors, no compute (mirrors cintx-core)
│   ├── pyscf-runtime/         # backend dispatch, planner, workspace pool (mirrors cintx-runtime)
│   ├── pyscf-kernels/         # ALL #[cube] kernels (mirrors xcfun-kernels role)
│   ├── pyscf-gto/             # Mole + basis + integral facade (bridges cintx)
│   ├── pyscf-scf/             # RHF/UHF/GHF + DIIS
│   ├── pyscf-dft/             # RKS/UKS + grids + libxc_rs + xcfun_rs glue
│   ├── pyscf-mp2/             # RMP2/UMP2 + ao2mo
│   ├── pyscf-ccsd/            # RCCSD/UCCSD
│   ├── pyscf-grad/            # analytical gradients for HF/DFT/MP2/CCSD
│   ├── pyscf-geomopt/         # BFGS driver
│   ├── pyscf-oracle/          # PySCF-as-live-oracle harness (pyo3 to upstream)
│   ├── pyscf-py/              # PyO3 cdylib — the wheel (mirrors xcfun-py)
│   ├── pyscf-capi/            # OPTIONAL C ABI shim (Phase >= v1.x; not v1)
│   └── pyscf-bench/           # criterion benchmarks (mirrors xcfun-rs benches)
├── xtask/                     # build helpers (mirrors all three siblings)
└── python/pyscf/              # python-source side of the wheel; re-exports _native
```

**Crate count vs siblings:** cintx=8, xcfun_rs=8, libxc_rs=27 (mostly kernel families), pyscf_rs=14 + xtask. Same order as the layered siblings.

**Deviations from the literal `pyscf-{module}-{rs,cubecl,oracle,py,runtime,compat}` shape:**
- **No per-method `*-cubecl` crate.** One `pyscf-kernels` instead. Rationale: §2.2.
- **No per-method `*-runtime` crate.** One `pyscf-runtime` instead. Same rationale.
- **No per-method `*-compat` crate.** The "compat" crate in cintx exists to bridge cubecl ↔ raw libcint C ABI; pyscf_rs has no equivalent legacy ABI to bridge. The Python-side compat (PySCF API surface preservation) lives entirely in `pyscf-py` + `python/pyscf/`.
- **No per-method `*-py` crate.** One `pyscf-py` exposes everything as one wheel. PyO3 0.28 supports submodules so `from pyscf import gto, scf, dft` still works.
- **No per-method `*-oracle` crate.** One `pyscf-oracle` covers all methods (PySCF is the oracle for everything).
- **No per-method `*-rs` crate.** The seven `pyscf-{method}` crates *are* the safe-facade crates; suffixing them `-rs` is redundant since the workspace name is already `pyscf_rs`.

**Risk of deviating:** sibling-crate muscle-memory expects "every method has every layer-suffix." Rationale must be explicit in `pyscf-rs/Cargo.toml` comments so future contributors don't try to "fix" it by splitting kernels per method.

---

## 3. Component Responsibilities (Cargo.toml-level)

| Crate | Owns | Dependencies (path) | Dependencies (external) |
|---|---|---|---|
| `pyscf-core` | `Mole`, `AOIntegrals` (handle), `MOIntegrals` (handle), `Density` (RDM1, RDM2), `MOCoefficients`, `MOEnergies`, `Amplitudes` (T1/T2), `Energy` (typed wrapper, units = Hartree), `Method` trait, `Density` trait, `Gradient` trait, `XcError`, all `thiserror` enums | none | `thiserror`, `smallvec`, `num-traits`, `nalgebra` (only `Matrix3<f64>` for geometry), `tracing` |
| `pyscf-runtime` | `BackendKind` enum (CPU/CUDA/WGPU/ROCm/Metal), `WorkspacePool`, `Planner`, `Scheduler`, `BatchHandle`, `DeviceHandle`, env-var parsing (PYSCF_TMPDIR, PYSCF_MAX_MEMORY, PYSCF_BACKEND) | `pyscf-core` | `tracing`, `cubecl` (no runtime deps — flags only enable enum arms; mirrors cintx-runtime) |
| `pyscf-kernels` | Every `#[cube]` body: 1e/2e Fock build, ao2mo, DFT numint (rho/vrho), CC tensor contractions, gradient kernels. Module-per-method internally. | `pyscf-core`, `pyscf-runtime` | `cubecl`, `cubecl-cpu` (default), `cubecl-cuda`/`cubecl-wgpu`/`cubecl-hip` (optional), `bytemuck` |
| `pyscf-gto` | `Mole::build`, basis-set loading from JSON, ECP, AO-integral request API (`mol.intor("int2e")` analog), GTO eval on grid points | `pyscf-core`, `pyscf-runtime`, `pyscf-kernels`, **`cintx`** (workspace path dep) | `serde`, `serde_json` |
| `pyscf-scf` | `RHF`, `UHF`, `GHF` structs, `kernel()` driver, `get_init_guess()`, `get_fock()`, `make_rdm1()`, `eig()`, DIIS via `pyscf-core::Diis` | `pyscf-core`, `pyscf-runtime`, `pyscf-kernels`, `pyscf-gto` | `tracing` |
| `pyscf-dft` | `RKS`, `UKS`, Becke grid + Lebedev points, `numint::nr_rks`, XC dispatch (`libxc_rs::Functional` or `xcfun_rs::Functional`) | `pyscf-core`, `pyscf-runtime`, `pyscf-kernels`, `pyscf-gto`, `pyscf-scf`, **`libxc_rs`**, **`xcfun_rs`** | — |
| `pyscf-mp2` | `RMP2`, `UMP2`, `DFRMP2`, AO→MO integral transformation (calls `pyscf-kernels::ao2mo`) | `pyscf-core`, `pyscf-runtime`, `pyscf-kernels`, `pyscf-gto`, `pyscf-scf` | — |
| `pyscf-ccsd` | `RCCSD`, `UCCSD`, T1/T2 amplitude solver, amplitude-DIIS | `pyscf-core`, `pyscf-runtime`, `pyscf-kernels`, `pyscf-gto`, `pyscf-scf`, `pyscf-mp2` | — |
| `pyscf-grad` | `Gradient<RHF>`, `Gradient<RKS>`, `Gradient<MP2>`, `Gradient<CCSD>`, CPHF solver, Λ-equation solver | `pyscf-core`, `pyscf-runtime`, `pyscf-kernels`, `pyscf-gto`, `pyscf-scf`, `pyscf-dft`, `pyscf-mp2`, `pyscf-ccsd` | — |
| `pyscf-geomopt` | BFGS driver, geometry step-controllers, convergence checks | `pyscf-core`, `pyscf-grad` | `argmin = "0.10"`, `argmin-math` |
| `pyscf-oracle` | `pyo3::Python::with_gil` harness that imports upstream `pyscf` and runs the same calculation, returning numbers diffable against ours | `pyscf-core` | `pyo3`, `numpy`, `serde_json`, `approx` (dev-dep only — see §6) |
| `pyscf-py` | `#[pymodule]` cdylib. Sub-modules: `_native.gto`, `_native.scf`, `_native.dft`, `_native.mp`, `_native.cc`, `_native.grad`, `_native.geomopt` | `pyscf-core`, `pyscf-gto`, `pyscf-scf`, `pyscf-dft`, `pyscf-mp2`, `pyscf-ccsd`, `pyscf-grad`, `pyscf-geomopt` | `pyo3 = "0.28"`, `numpy = "0.28"` (mirror xcfun-py) |
| `pyscf-bench` | criterion benchmarks (RHF/water, RKS/biphenyl, MP2/glycine, …) | every method crate | `criterion` |
| `pyscf-capi` (deferred) | Optional C ABI for non-Python consumers | `pyscf-core`, every method crate | — |

---

## 4. Strict-DAG dependency graph

```
                        ┌──────────────┐
                        │  pyscf-core  │   (types, traits, errors)
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │pyscf-runtime │   (backend dispatch, no cubecl deps)
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │pyscf-kernels │   (all #[cube] bodies)
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  pyscf-gto   │ ──► cintx (external workspace)
                        └──────┬───────┘
                               │
                        ┌──────▼───────┐
                        │  pyscf-scf   │
                        └──────┬───────┘
              ┌────────────────┼────────────────┐
              │                │                │
       ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐
       │  pyscf-dft  │  │  pyscf-mp2  │  │             │
       └──────┬──────┘  └──────┬──────┘  │             │
              │                │         │             │
              │         ┌──────▼──────┐  │             │
              │         │ pyscf-ccsd  │  │             │
              │         └──────┬──────┘  │             │
              │                │         │             │
              └────────────────┴─────────┴─────┐       │
                                               │       │
                                        ┌──────▼───────▼──┐
                                        │   pyscf-grad    │
                                        └────────┬────────┘
                                                 │
                                        ┌────────▼────────┐
                                        │ pyscf-geomopt   │
                                        └────────┬────────┘
                                                 │
       (every method crate)                      │
              │                                  │
              └─►─►─►─►─►─►─►─►─►─►─►─►─►─►─►─►─►┤
                                                 │
                                          ┌──────▼──────┐
                                          │  pyscf-py   │  (one wheel)
                                          └─────────────┘

  pyscf-oracle ──► (upstream PySCF via pyo3)   [parallel; dev-only path]
  pyscf-bench  ──► (every method crate)        [parallel; dev-only path]
  pyscf-rs (workspace root) re-exports pyscf-{gto,scf,dft,mp2,ccsd,grad,geomopt}
```

**Acyclicity check:**
- `pyscf-grad` is the only crate that depends on multiple methods, but it never receives a back-edge (no method depends on `pyscf-grad`).
- `pyscf-geomopt` only depends on `pyscf-grad` (which already pulls everything else).
- `pyscf-ccsd → pyscf-mp2` is forward (CCSD warm-starts from MP2 amplitudes).
- `pyscf-py` is downstream of every method crate — no method imports `pyscf-py`.

**Critical-path build order** (Phase 1 of each box must complete before Phase 2 of any downstream):

| Wave | Crates buildable in parallel | Why |
|---|---|---|
| W0 | `pyscf-core` | Foundation — no internal deps. |
| W1 | `pyscf-runtime` | Depends only on core. |
| W2 | `pyscf-kernels` | Depends on core + runtime. CUBECL build cost lives here — biggest single crate. |
| W3 | `pyscf-gto` | Pulls `cintx` (assumed already built in sibling workspace). |
| W4 | `pyscf-scf` | The fork point — DFT/MP2/CCSD/grad all depend on SCF. |
| W5 | `pyscf-dft`, `pyscf-mp2` | Parallel — neither depends on the other. |
| W6 | `pyscf-ccsd` | Depends on MP2 (warm start). |
| W7 | `pyscf-grad` | Depends on all methods. |
| W8 | `pyscf-geomopt`, `pyscf-py` | Parallel — both leaf consumers. |
| Wx | `pyscf-oracle`, `pyscf-bench` | Anywhere; dev-graph only. |

---

## 5. cubecl integration pattern

**Decision:** one `pyscf-kernels` crate; backend selection by feature flag (mirrors `cintx-cubecl`/`xcfun-kernels`/`xcfun-gpu`).

### 5.1 Feature matrix on `pyscf-kernels/Cargo.toml`

```toml
[features]
default = ["cpu"]
cpu   = ["dep:cubecl-cpu",  "pyscf-runtime/cpu"]
cuda  = ["dep:cubecl-cuda", "pyscf-runtime/cuda"]
wgpu  = ["dep:cubecl-wgpu", "pyscf-runtime/wgpu"]
rocm  = ["dep:cubecl-hip",  "pyscf-runtime/rocm"]
metal = ["wgpu",            "pyscf-runtime/metal"]   # alias; cubecl-metal does not ship
# Per-method kernel gates so pyscf-scf-only builds skip CCSD/MP2 kernels.
with-ccsd = []
with-mp2  = []
with-grad = []
```

This **forwards backend features down to `pyscf-runtime`** (the typed-enum-arm crate) — *exactly* what cintx does (`cintx-cubecl/Cargo.toml:20-29`).

### 5.2 Where runtime selection lives

Same place as cintx: in `pyscf-runtime`. `BackendKind` is the typed enum, `auto_backend()` walks an env-var → autodetect priority chain (`PYSCF_BACKEND=cuda` overrides; otherwise probe-order is `cuda → rocm → metal → wgpu → cpu`). xcfun-gpu's `auto_backend` (`crates/xcfun-gpu/src/`) is the reference implementation.

### 5.3 What does NOT go in `pyscf-kernels`

- No host-side state (`Mole`, density matrices, MO coefficients) — that's `pyscf-core`.
- No driver loops (SCF iteration, DIIS, BFGS) — that's the method crates.
- No PySCF API mirroring — that's `pyscf-py` + `python/pyscf/`.

The kernels crate is **pure GPU/SIMD primitive code** plus the dispatch table that selects which kernel + workgroup-size to launch given a `BackendKind`.

---

## 6. PyO3 layer: single `pyscf-py` crate with submodules

**Decision:** one cdylib (mirrors `xcfun-py`), exposes a `_native` extension module with PyO3 0.28 sub-modules. The user-facing `pyscf` package is a Python source dir at `python/pyscf/` that re-exports from `_native`.

### 6.1 Wheel layout

```
python/pyscf/                       # the user-visible package
  __init__.py                       # imports from _native; preserves PYSCF_EXT_PATH plugin loader
  __config__.py                     # PYSCF_MAX_MEMORY, PYSCF_TMPDIR defaults
  gto/__init__.py                   # from pyscf._native.gto import M, Mole, …
  scf/__init__.py                   # from pyscf._native.scf import RHF, UHF, GHF
  dft/__init__.py                   # from pyscf._native.dft import RKS, UKS
  mp/__init__.py                    # from pyscf._native.mp import MP2 / RMP2 / UMP2
  cc/__init__.py                    # from pyscf._native.cc import CCSD / RCCSD / UCCSD
  grad/__init__.py
  geomopt/__init__.py
  lib/                              # checkpoint I/O, logger shim — pure Python
  _native.so                        # the cdylib produced by pyscf-py
```

This preserves `from pyscf import gto, scf, dft` exactly. The Python `__init__.py` files are tiny re-exports.

### 6.2 PyO3 submodule wiring

In `pyscf-py/src/lib.rs`:

```rust
#[pymodule]
fn _native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let gto_mod = PyModule::new_bound(py, "gto")?;
    crate::gto::register(&gto_mod)?;
    m.add_submodule(&gto_mod)?;

    let scf_mod = PyModule::new_bound(py, "scf")?;
    crate::scf::register(&scf_mod)?;
    m.add_submodule(&scf_mod)?;

    // … dft, mp, cc, grad, geomopt
    Ok(())
}
```

Each `crate::{method}::register` lives in `pyscf-py/src/{method}.rs` and adds the relevant `pyclass`/`pyfunction`s to the submodule. All seven submodules in one cdylib — same as how scipy bundles `scipy.linalg`, `scipy.optimize`, etc.

### 6.3 Why not seven separate `*-py` crates

- **One wheel = one `pip install`.** Splitting would force users to install seven packages or ship a meta-package, defeating the "drop-in" requirement.
- **Cross-method types (e.g. `Mole`) must be the same `pyclass` across submodules.** If `pyscf-gto-py` and `pyscf-scf-py` were separate cdylibs, each would define its own `Mole` pyclass and they'd be type-incompatible.
- **xcfun-py proves the single-cdylib pattern works in this family** (`crates/xcfun-py/Cargo.toml:9-10`).

### 6.4 Backend feature plumbing in `pyscf-py`

Mirror xcfun-py exactly (`crates/xcfun-py/Cargo.toml:14-26`): `pyscf-py` never depends on `cubecl-*` directly — it forwards `cpu/cuda/wgpu/rocm/metal` features through the method crates, which forward to `pyscf-runtime` and `pyscf-kernels`.

---

## 7. Oracle integration

**Decision:** dedicated `pyscf-oracle` workspace crate, used as a `dev-dependencies = { ... }` from each method crate's integration tests. The oracle is *not* a runtime feature flag.

### 7.1 What the oracle does

```rust
// pyscf-oracle/src/lib.rs
pub fn pyscf_rhf_oracle(geometry: &str, basis: &str) -> OracleResult {
    Python::with_gil(|py| {
        let pyscf = py.import_bound("pyscf")?;
        let mol = pyscf.getattr("M")?.call((), Some(&kwargs! { atom: geometry, basis: basis }))?;
        let mf = pyscf.getattr("scf")?.getattr("RHF")?.call1((mol,))?;
        mf.call_method0("kernel")?;
        Ok(OracleResult {
            e_tot: mf.getattr("e_tot")?.extract::<f64>()?,
            mo_coeff: mf.getattr("mo_coeff")?.extract::<Vec<Vec<f64>>>()?,
            mo_energy: mf.getattr("mo_energy")?.extract::<Vec<f64>>()?,
            converged: mf.getattr("converged")?.extract::<bool>()?,
        })
    })
}
```

Each method crate's `tests/oracle_*.rs` runs both the Rust kernel and the oracle in the same process and asserts agreement to 1e-9 Hartree (or method-specific tolerance).

### 7.2 Gating

- `pyscf-oracle` is a **regular library crate** (not a feature flag) but its `[dependencies]` includes `pyo3 = { version = "0.28", features = ["auto-initialize"] }`.
- **Test crates that need it list it as `dev-dependencies`**, *not* `dependencies`. Method crates' release builds never link Python.
- CI can disable oracle tests on a runner without PySCF installed via `--exclude pyscf-oracle` or a `pyscf-oracle/disabled` no-op feature.
- This pattern works because `pyo3` with `auto-initialize` will discover the system Python and import upstream `pyscf` from `site-packages`. The repo's vendored `pyscf/` directory at the workspace root is on `PYTHONPATH` via a `.cargo/config.toml` env var.

### 7.3 Tolerance policy

```toml
# pyscf-oracle/Cargo.toml
[package.metadata.pyscf-oracle]
energy_tolerance_hartree     = 1e-10   # bit-exact target
energy_tolerance_relaxed_dft = 1e-8    # numerical grid integration
gradient_tolerance_au        = 1e-7
```

Each test specifies its tolerance via a `oracle_check!(method, tol)` macro defined in `pyscf-oracle`.

---

## 8. Data flow

### 8.1 Single-point energy calculation

```
User Python                 pyscf-py            pyscf-{method}      pyscf-kernels      pyscf-runtime
─────────────────────────────────────────────────────────────────────────────────────────────────
gto.M(atom=..., basis=...)
        │ ─────────────────► gto::M_py
        │                          │ ──────────► Mole::build (gto)
        │                          │                  │ ─────► cintx (basis tabs)
        │                          │                  │            └────► Mole = (atoms, _atm, _bas, _env)
        │ ◄──── PyMole ─────────────────────── (via Bound<Mole>)
        │
mf = scf.RHF(mol)
        │ ─────────────────► scf::RHF::__new__
        │                          │ ──────────► RHF::new(&Mole) ─► RhfState (no compute yet)
        │ ◄──── PyRHF ──────────────
        │
mf.kernel()
        │ ─────────────────► RHF.kernel_py
        │                          │ ──────────► RHF::kernel
        │                          │                  │ get_hcore   ─────► kernels::int1e         ─►  runtime::dispatch ─► CUBECL
        │                          │                  │ get_init_guess
        │                          │                  │ loop:
        │                          │                  │   get_fock  ─────► kernels::vhf::jk_build ─►  runtime::dispatch ─► CUBECL
        │                          │                  │   eig (LAPACK or cubecl-eig)
        │                          │                  │   make_rdm1
        │                          │                  │   diis.update
        │                          │                  │ converged?
        │                          │                  └► returns Energy + MOCoefficients
        │ ◄──── e_tot ───────────────
```

### 8.2 Post-SCF stack

```
mf : RHF (converged)            ─► provides MOCoefficients + MOEnergies
   │
   ├── DFT goes here too: KohnSham composes RHF behavior + numint
   │
   ├── pt = mp.MP2(mf).kernel()
   │       │ ao2mo: AOIntegrals → MOIntegrals (kernels::ao2mo)
   │       │ assemble denominator, build amplitudes, compute E_corr
   │       └► returns Amplitudes (T2) + Energy
   │
   ├── cc = cc.CCSD(mf).kernel()
   │       │ ao2mo same as above
   │       │ initialize T1=0, T2 from MP2 (via pyscf-mp2)
   │       │ amplitude loop (T1, T2 update + DIIS)
   │       └► returns Amplitudes (T1, T2) + Energy
   │
   └── grad.RHF(mf).kernel()
           │ build derivative integrals (kernels::int1e_ip, int2e_ip)
           │ solve CPHF (response equations) — needs J/K from kernels::vhf
           └► returns 3N gradient array
```

### 8.3 Major data structures (which crate owns each)

| Data structure | Owner crate | Approx. shape |
|---|---|---|
| `Mole` | `pyscf-core` (struct) / built by `pyscf-gto` | `{atoms: Vec<Atom>, _atm, _bas, _env: Vec<f64>, basis: BasisSet, ecp: Option<ECP>, charge, spin, symmetry}` |
| `BasisSet` | `pyscf-core` | re-exports `cintx_core::BasisSet` to keep AO indexing consistent across crates |
| `AOIntegrals` (handle) | `pyscf-core` | opaque handle to a kernel-launch plan; values live on device |
| `MOIntegrals` (handle) | `pyscf-core` | same; produced by `kernels::ao2mo` |
| `Density` (RDM1) | `pyscf-core` | `Array2<f64>` for restricted; `(Array2, Array2)` for unrestricted |
| `RDM2` | `pyscf-core` | tensor handle (full 4-index too large for in-core typically) |
| `MOCoefficients` | `pyscf-core` | `Array2<f64>` AO×MO |
| `MOEnergies` | `pyscf-core` | `Array1<f64>` |
| `Amplitudes<T1>` `Amplitudes<T2>` | `pyscf-core` (generic over rank) | tensors on device |
| `Energy` | `pyscf-core` | `f64` newtype with units (Hartree); `impl ops` for delta arithmetic |
| `Gradient` | `pyscf-core` | `Array2<f64>` 3×N |
| `Hessian` | `pyscf-core` (out of scope for v1) | — |
| `BackendKind` | `pyscf-runtime` | enum CPU / CUDA / WGPU / ROCm / Metal |
| `WorkspacePool` | `pyscf-runtime` | per-device `OnceLock<DeviceClient>` cache |

### 8.4 Storage policy: where do tensors live?

- **Small/per-iteration data on host:** Mole metadata, MO coefficients (≤ 10k×10k = 800 MB; usually much smaller), DIIS error vectors.
- **Large tensors on device:** AO integrals (4-index, never materialized; built into J/K on the fly), MO 2e integrals (`(ov|ov)`-shaped for MP2/CCSD), CCSD T2 amplitudes (stored on device when possible, spilled to chkfile if MAX_MEMORY exceeded).
- **Spill mechanism:** `pyscf-runtime::WorkspacePool` queries available device memory; if the request exceeds `PYSCF_MAX_MEMORY`, it falls back to a chkfile-backed allocator (HDF5).

---

## 9. Trait design — the major abstractions

These are sketches, not final signatures. Each lives in `pyscf-core`.

```rust
// pyscf-core/src/method.rs
pub trait Method {
    type State;
    type Output;

    /// Drive the method to completion. Returns owned result.
    fn kernel(&mut self) -> Result<Self::Output, MethodError>;

    /// Reference to converged state for downstream consumers.
    fn state(&self) -> &Self::State;

    /// Set max iterations / tolerance / verbosity.
    fn with_options(self, opts: MethodOptions) -> Self;
}

// pyscf-core/src/scf.rs — additional contract for SCF-like methods
pub trait Scf: Method<Output = ScfResult> {
    fn get_hcore(&self) -> &Hcore;
    fn get_fock(&self, dm: &Density) -> Result<Fock, MethodError>;
    fn get_init_guess(&self) -> Density;
    fn make_rdm1(&self, mo_coeff: &MOCoefficients, mo_occ: &MOOccupation) -> Density;
    fn eig(&self, fock: &Fock, s: &Overlap) -> (MOEnergies, MOCoefficients);
    fn energy_tot(&self, dm: &Density) -> Energy;
}

// pyscf-core/src/dft.rs — KS extends SCF
pub trait KohnSham: Scf {
    fn xc_functional(&self) -> &dyn XcFunctional;
    fn grids(&self) -> &Grid;
    fn get_veff(&self, dm: &Density) -> Result<Veff, MethodError>;
}

// pyscf-core/src/post_scf.rs
pub trait PostScf<'a> {
    type Reference: Scf;
    type Output;

    fn from_reference(reference: &'a Self::Reference) -> Self;
    fn kernel(&mut self) -> Result<Self::Output, MethodError>;
}

// pyscf-core/src/grad.rs
pub trait Gradient {
    type Method: Method;
    fn kernel(&self, m: &Self::Method) -> Result<GradientResult, MethodError>;
}

// pyscf-core/src/density.rs — open for plug-in spin types
pub trait DensityRepr {
    fn shape(&self) -> (usize, usize);
    fn n_electrons(&self) -> f64;
    fn trace_with(&self, other: &Self) -> f64;
}

impl DensityRepr for RDM1Restricted { … }
impl DensityRepr for RDM1Unrestricted { … }
impl DensityRepr for RDM1Generalized { … }

// pyscf-core/src/xc.rs — XC functional plug point
pub trait XcFunctional: Send + Sync {
    fn id(&self) -> &str;
    fn family(&self) -> XcFamily;             // LDA / GGA / mGGA / hybrid
    fn hyb_coeff(&self) -> f64;                // exact-exchange fraction
    fn eval(&self, rho: &RhoBatch, mode: XcMode) -> Result<XcOutput, XcError>;
}

impl XcFunctional for libxc_rs::Functional { … }
impl XcFunctional for xcfun_rs::Functional { … }

// pyscf-core/src/integrator.rs — integral plug point
pub trait IntegralEngine: Send + Sync {
    fn int1e(&self, mol: &Mole, op: Operator1e) -> Result<AOIntegralBatch, IntegralError>;
    fn int2e(&self, mol: &Mole) -> Result<AOIntegralBatch, IntegralError>;
    fn intor_deriv(&self, mol: &Mole, op: Operator1e, order: usize)
        -> Result<AOIntegralBatch, IntegralError>;
}

// Default impl wraps cintx::SessionRequest. Users could swap a custom engine.
```

**Plug points:** `XcFunctional`, `IntegralEngine`, `BasisLoader` (for custom basis-set sources), `Diis` (the DIIS algorithm — users could swap for ADIIS).

---

## 10. Patterns

### Pattern 1 — Horizontal layered façade with cubecl in the middle (mirrors cintx + xcfun_rs)

**What:** `core → runtime → kernels → method crates → py/capi/oracle` with backend feature flags forwarded down each step.

**When to use:** any cubecl-backed compute library in this family.

**Trade-offs:** + crystal-clear ownership; + each layer compiles independently; + parallelizable build past wave 2; − seven shared layer-crates means even the smallest method drag-in compiles `pyscf-core/runtime/kernels`.

### Pattern 2 — Method-state struct with a `kernel()` driver, not inheritance

**What:** PySCF uses Python class inheritance (`RHF` ← `SCF` ← `StreamObject`). Rust port uses **composition + traits**: `RhfState { mol: Arc<Mole>, opts: ScfOptions, … }` + `impl Scf for RhfState`. A `KohnSham<RhfState>` wraps an SCF state with a grid + functional rather than inheriting from it.

**When to use:** every method that PySCF expressed as a class hierarchy.

**Trade-offs:** + no inheritance footguns; + cleaner ownership of `&Mole`; + supports zero-cost specialization via `impl Scf for KohnSham<RhfState>`; − contributors used to PySCF will reach for inheritance and need style-guide nudges.

### Pattern 3 — Backend selection by typed-enum + feature flag (cintx-runtime convention)

**What:** `BackendKind::{Cpu, Cuda, Wgpu, Rocm, Metal}` lives in `pyscf-runtime`. Each variant is `#[cfg(feature = "...")]`. The kernels crate dispatches on `BackendKind` at runtime; flags only enable the typed arm.

**When to use:** every place where you'd be tempted to take `<R: Runtime>` as a generic.

**Trade-offs:** + concrete error messages when a backend is missing at runtime; + same binary can support multiple backends; − every kernel-launch site has a 5-arm match (or a macro).

### Pattern 4 — Oracle as a regular dev-dependency, not a feature flag

**What:** `pyscf-oracle` is a normal crate that exposes `oracle_rhf(...)`/`oracle_rks(...)`/etc. functions. Method crates' integration tests `dev-dependencies` it; release builds never link `pyo3`.

**When to use:** numerical-regression-test suites where the reference is itself a runnable program (here: upstream PySCF).

**Trade-offs:** + zero release-binary cost; + oracle calls run in-process so they're fast; − requires Python at test time; − tests must serialize Python access (single GIL).

### Pattern 5 — Single `pyscf-py` cdylib with PyO3 submodules

**What:** one wheel contains `_native.gto`, `_native.scf`, etc. The user-visible `pyscf` package is a Python source dir that re-exports.

**When to use:** any compat layer for an existing multi-namespace Python library.

**Trade-offs:** + cross-method `pyclass`es (`Mole`) are type-compatible; + one `pip install`; − the cdylib gets large (≥ all kernels statically linked); − a single PyO3 ABI version locks the entire wheel.

### Pattern 6 — Kernel-feature gating to keep small builds small

**What:** `pyscf-kernels` ships features `with-mp2`, `with-ccsd`, `with-grad`, `with-dft`. The method crates set `pyscf-kernels = { features = ["with-mp2"] }` etc. A library consumer only doing HF gets no MP2/CCSD kernel code compiled.

**When to use:** when one giant kernels crate would otherwise force every consumer to compile every kernel.

**Trade-offs:** + dramatic cold-build improvements for SCF-only users; − feature-flag matrix complexity; − requires `cargo +nightly check --features=…` matrix in CI.

---

## 11. Configuration / I/O

### 11.1 Environment variables (mirror PySCF)

| Var | Owner crate | Effect |
|---|---|---|
| `PYSCF_TMPDIR` | `pyscf-runtime` | scratch directory for HDF5 spill / chkfile defaults |
| `PYSCF_MAX_MEMORY` | `pyscf-runtime` | host memory ceiling for in-core / direct-SCF decisions |
| `PYSCF_BACKEND` | `pyscf-runtime` | force backend (`cpu/cuda/wgpu/rocm/metal`) |
| `PYSCF_EXT_PATH` | `pyscf-py` (Python init) | namespace-package plugin discovery (Python-side only) |
| `RUST_LOG` | every crate | `tracing` filter |

### 11.2 Checkpoint files (HDF5 / chkfile compat)

- Library: `hdf5 = "0.8"` (or `hdf5-metno` if maintainer churn matters; both wrap the same C lib).
- Layout: must match upstream PySCF's `chkfile` exactly (`/scf/e_tot`, `/scf/mo_coeff`, `/scf/mo_energy`, `/scf/mo_occ`, `/mol`).
- Owner: `pyscf-core::chkfile` module, used by every method crate.
- Roundtrip oracle test: write chkfile from Rust → read with upstream PySCF → assert numerical match (and vice-versa).

### 11.3 Logging

- `tracing` everywhere (mirrors all three sibling crates).
- `pyscf-py` installs a `tracing-subscriber` that maps `tracing::info!` → Python `logging.info`, preserving PySCF's verbosity-level conventions (`mol.verbose = 4` → `RUST_LOG=pyscf=debug`).
- Per-object `verbose` attribute is preserved on `pyclass`es for API compatibility but routes through `tracing::Level`.

### 11.4 Configuration object

Mirrors PySCF's `__config__.py`:

```rust
// pyscf-core/src/config.rs
pub struct GlobalConfig {
    pub max_memory_mb: usize,
    pub tmpdir: PathBuf,
    pub backend: BackendKind,
    pub default_basis: String,        // "STO-3G"
    pub default_xc: String,           // "PBE"
    pub verbosity: tracing::Level,
}

pub static CONFIG: OnceLock<GlobalConfig> = OnceLock::new();
```

Loaded at first access from env vars; CLI / Python side can override before first kernel call.

---

## 12. Anti-patterns — what NOT to do

### Anti-pattern 1: per-method `*-cubecl` crates

**What people do:** literal reading of "mirror sibling pattern" → 7 × 6 crates.

**Why wrong:** cubecl kernels straddle methods (vhf is shared between SCF and DFT; ao2mo between MP2/CCSD/grad). Splitting forces either duplication or cross-method `*-cubecl` deps that break the DAG.

**Do this instead:** one `pyscf-kernels` crate with method-gated features.

### Anti-pattern 2: making `pyscf-grad` re-implement every method

**What people do:** because gradients need J/K, ao2mo, T2 amplitudes, etc., it's tempting to re-implement them inside `pyscf-grad`.

**Why wrong:** doubles the maintenance burden and creates two sources of truth.

**Do this instead:** `pyscf-grad` depends on every method it differentiates and *calls* their public API. Derivative integrals are new kernels in `pyscf-kernels::grad`, but the orchestration (CPHF solver) reuses `pyscf-scf::Diis`, `pyscf-mp2::ao2mo`, etc.

### Anti-pattern 3: oracle as a runtime feature

**What people do:** add `feature = "oracle"` to every method crate that pulls `pyo3`.

**Why wrong:** release builds end up linking Python; users without Python can't use the Rust crate; cross-compilation is harder.

**Do this instead:** §7 — oracle is a normal crate, listed in `dev-dependencies` only.

### Anti-pattern 4: putting `Mole` in `pyscf-gto`

**What people do:** PySCF puts `Mole` in `pyscf/gto/mole.py` so the Rust port mirrors that.

**Why wrong:** `pyscf-scf`, `pyscf-dft`, `pyscf-mp2` all need `Mole` as a struct. If it lives in `pyscf-gto`, every method crate depends on `pyscf-gto`. That's actually fine *until* you want to run an MP2 calculation in a test that doesn't construct Mole through the gto facade — circular convenience.

**Do this instead:** `Mole` lives in `pyscf-core`. `pyscf-gto` provides the `Mole::build()` builder + basis loader + integral engine. Other methods only need the struct.

### Anti-pattern 5: Python-side method dispatch by class lookup

**What people do:** Python-side `from pyscf import scf; scf.RHF` does dynamic class lookup in the `pyscf.scf` module.

**Why wrong:** when `pyscf-py` is a single cdylib with PyO3 submodules, every `pyclass` is registered eagerly. Fine.

**Do this instead:** keep the lazy-loading shim in `python/pyscf/__init__.py` for symmetry with PySCF's plugin discovery (`PYSCF_EXT_PATH`), but route every actual class through the cdylib eagerly.

### Anti-pattern 6: shared mutable Mole across SCF iterations

**What people do:** `RHF.kernel()` mutates `mol._env` in place during iteration.

**Why wrong:** breaks `Send`/`Sync`; PyO3 borrow tracker rejects mutation while a `Bound<Mole>` exists.

**Do this instead:** `Mole` is `Arc<MoleInner>` with interior immutability after build. Per-iteration scratch lives in `RhfState`, not on `Mole`.

---

## 13. Scaling considerations (compute scale, not user scale)

| Scale | Adjustments |
|---|---|
| < 100 atoms (small organics) | All in-core; CPU SIMD via cubecl-cpu often beats single-GPU for setup-heavy kernels; default backend selection. |
| 100–500 atoms (medium proteins) | Direct SCF mandatory (no integral storage); CUDA/HIP wins; spill T2 amplitudes to chkfile if MAX_MEMORY exceeded; density fitting (DF-RHF, DF-MP2) becomes worth it. |
| 500–2000 atoms | Beyond v1 scope but architecture should not preclude: Resolution-of-Identity ao2mo, frozen-core CCSD, multi-GPU (later milestone), exchange screening. |
| > 2000 atoms | PBC, fragment methods, MPI distribution — explicitly Out of Scope. |

**First bottleneck likely to hit:** `pyscf-kernels::ao2mo` for MP2 / CCSD on > 200 atoms. Mitigate by introducing density fitting (`pyscf-mp2/Cargo.toml feature = "df"`) before naive O(N⁵) MP2 saturates memory.

**Second bottleneck:** DFT grid integration on large systems. Mitigate by Becke-grid pruning + BLYP-style atomic-grid radial-shell skipping.

---

## 14. Integration points

### 14.1 External Rust dependencies (path deps to sibling workspaces)

| Dep | Used by | Why |
|---|---|---|
| `cintx` (workspace `~/Documents/workspace/cintx`) | `pyscf-gto` | AO integrals (`SessionRequest`, `BasisSet`, `Operator`). Path dep, NOT crates.io, until cintx is published. |
| `libxc_rs` (workspace `~/Documents/workspace/libxc_rs`) | `pyscf-dft` | LDA/GGA/mGGA functional values + first/second derivatives. |
| `xcfun_rs` (workspace `~/Documents/workspace/xcfun_rs`) | `pyscf-dft` | Higher-order analytic XC derivatives (needed for analytic CPKS / Hessian; v1 grad uses second derivatives). |

These appear in `pyscf-rs/Cargo.toml` as:

```toml
[workspace.dependencies]
cintx     = { path = "../cintx",     default-features = false }
libxc_rs  = { path = "../libxc_rs",  default-features = false }
xcfun-rs  = { path = "../xcfun_rs/crates/xcfun-rs",  default-features = false }
```

### 14.2 External C dependencies — none

- libcint: replaced by `cintx`.
- libxc: replaced by `libxc_rs`.
- xcfun: replaced by `xcfun_rs`.
- BLAS/LAPACK for dense eig: handled via `cubecl-cpu` (which uses `linalg` kernels) or `nalgebra-lapack`/`faer` for the small SCF eigensolver. Decision deferred to STACK.md research.
- HDF5: `hdf5` crate (Rust binding to libhdf5).

### 14.3 Internal crate boundaries

| Boundary | Communication | Notes |
|---|---|---|
| `pyscf-py` ↔ `pyscf-{method}` | direct Rust function calls inside cdylib | one-way (Python wraps Rust); no callbacks back into Python on hot paths. |
| `pyscf-{method}` ↔ `pyscf-kernels` | typed batch handles + `BackendKind` dispatch | hot path; no allocations. |
| `pyscf-kernels` ↔ cubecl runtime | cubecl `Client<R>` per backend | `OnceLock` cached. |
| `pyscf-gto` ↔ `cintx` | `cintx::SessionRequest` / `IntegralTensor` | thin wrapper; we don't shadow cintx types. |
| `pyscf-dft` ↔ `libxc_rs` / `xcfun_rs` | through `dyn XcFunctional` trait object | runtime-pluggable; LDA defaults to libxc, mGGA-derivatives default to xcfun. |
| `pyscf-oracle` ↔ upstream PySCF | `pyo3::Python::with_gil` per oracle call | dev-only. |

---

## 15. Build-order implications for the roadmap

**Critical path** (must be sequential):

1. `pyscf-core` — types, traits, errors, `Mole` struct, `Energy` newtype, chkfile schema.
2. `pyscf-runtime` — `BackendKind`, env-var config, workspace pool.
3. `pyscf-kernels` foundation — int1e CPU kernel, vhf CPU kernel, dispatch table; CUDA backend can land in a *later* phase.
4. `pyscf-gto` — Mole builder, basis loader, glue to `cintx`.
5. `pyscf-scf` (RHF only) — first end-to-end milestone (`gto.M(H2O) → scf.RHF.kernel() → e_tot` matches oracle).
6. **Fork point.** From here, `pyscf-dft` / `pyscf-mp2` / `pyscf-ccsd` are parallelizable.

**Parallelizable branches** (after critical-path step 5):

- DFT branch: `pyscf-dft` (RKS) → exercise `libxc_rs` / `xcfun_rs` integration.
- MP2 branch: `pyscf-mp2` (RMP2) → exercise ao2mo kernels.
- UHF / GHF: extend `pyscf-scf` in parallel with DFT.
- CCSD: serially after MP2 (warm-start dependency).

**Suggested phase ordering for the roadmap:**

| Phase | Crates created/extended | Milestone |
|---|---|---|
| 01 — Foundation | `pyscf-core`, `pyscf-runtime`, `pyscf-bench`, top-level workspace | `cargo build --workspace` succeeds; `Mole::build` works for H₂O. |
| 02 — Kernels & GTO | `pyscf-kernels` (int1e, int2e on CPU), `pyscf-gto`, `pyscf-oracle` skeleton | AO 1e integrals match cintx oracle to 1e-12 for H₂O / cc-pVDZ. |
| 03 — RHF | `pyscf-scf` (RHF + DIIS), `pyscf-py` skeleton | `pyscf.scf.RHF(mol).kernel()` matches upstream PySCF to 1e-9 Hartree on test set. |
| 04 — UHF + GHF | `pyscf-scf` extension | Open-shell oracle parity. |
| 05 — RKS | `pyscf-dft` (numint, grids, libxc_rs), grid kernels in `pyscf-kernels` | `pyscf.dft.RKS(mol, xc='PBE').kernel()` matches oracle. |
| 06 — UKS | `pyscf-dft` extension | Spin-polarized DFT. |
| 07 — RMP2 | `pyscf-mp2`, ao2mo kernels | RMP2 oracle parity. |
| 08 — UMP2 + DF-MP2 | `pyscf-mp2` extension | Density-fitted MP2 path. |
| 09 — RCCSD | `pyscf-ccsd`, T2 amplitude kernels | RCCSD oracle parity. |
| 10 — UCCSD | `pyscf-ccsd` extension | Unrestricted CCSD. |
| 11 — Gradients (HF + DFT) | `pyscf-grad`, derivative kernels in `pyscf-kernels`, CPHF/CPKS | `grad.RHF(mf).kernel()` matches oracle force vectors. |
| 12 — Gradients (MP2 + CCSD) | `pyscf-grad` extension, Λ-equation solver | post-SCF gradients. |
| 13 — Geomopt | `pyscf-geomopt` | Optimize H₂O geometry to PySCF reference structure. |
| 14 — GPU backends | enable `cuda`, `wgpu`, `rocm` features in `pyscf-kernels` + `pyscf-runtime`; per-backend regression suite | RHF/RKS run on CUDA, results match CPU within tolerance; speedup ≥ 2× target. |
| 15 — PyPI wheel | `pyscf-py` polish; `python/pyscf/` shim; maturin CI; cibuildwheel | wheel installs via `pip install pyscf-rs` and runs the upstream PySCF test snippet. |

The "GPU backends" phase intentionally lands **after** all methods work on CPU, mirroring how xcfun_rs deferred GPU to Plans 06-03/06-04 only after the cubecl-cpu substrate was solid.

---

## 16. Workspace skeleton — `Cargo.toml` files the roadmapper can implement directly

### 16.1 Top-level `Cargo.toml`

```toml
[workspace]
members = [
    "crates/pyscf-core",
    "crates/pyscf-runtime",
    "crates/pyscf-kernels",
    "crates/pyscf-gto",
    "crates/pyscf-scf",
    "crates/pyscf-dft",
    "crates/pyscf-mp2",
    "crates/pyscf-ccsd",
    "crates/pyscf-grad",
    "crates/pyscf-geomopt",
    "crates/pyscf-oracle",
    "crates/pyscf-py",
    "crates/pyscf-bench",
    "xtask",
]
default-members = []   # consumers pick crates; see cintx tech-debt note
resolver = "3"

[workspace.package]
version = "0.1.0"
edition = "2024"
rust-version = "1.92"   # match xcfun_rs (which pins cubecl 0.10.0)
license = "Apache-2.0"  # matches upstream PySCF
repository = "https://github.com/<tbd>/pyscf_rs"

[workspace.dependencies]
# cubecl pivot — must move in lockstep (xcfun_rs CLAUDE.md note)
cubecl       = "=0.10.0"
cubecl-cpu   = "=0.10.0"
cubecl-cuda  = "=0.10.0"
cubecl-wgpu  = "=0.10.0"
cubecl-hip   = "=0.10.0"

# Sibling-workspace path deps
cintx     = { path = "../cintx",                       default-features = false }
libxc_rs  = { path = "../libxc_rs",                    default-features = false }
xcfun-rs  = { path = "../xcfun_rs/crates/xcfun-rs",    default-features = false }

# PyO3 stack — pin to xcfun_rs's tested version
pyo3   = { version = "=0.28.3", default-features = false }
numpy  = { version = "=0.28.0" }

# Library deps
thiserror = "=2.0.18"
smallvec  = "1.13"
bytemuck  = { version = "1.25", features = ["derive"] }
tracing   = { version = "=0.1.44", default-features = false }
ndarray   = "0.16"
hdf5      = "0.8"
nalgebra  = "0.33"

# App-boundary
anyhow = "1.0"

# Dev / test / bench
approx     = "=0.5.1"
proptest   = "=1.11.0"
rstest     = "=0.26.1"
criterion  = { version = "=0.8.2", default-features = false, features = ["html_reports"] }
serde      = { version = "=1.0",   features = ["derive"] }
serde_json = "=1.0.149"

# Build-script helper
cc = { version = "1.2", features = ["parallel"] }

# Geomopt (only used by pyscf-geomopt)
argmin      = "0.10"
argmin-math = "0.4"

# Build-cost mitigations from libxc_rs (CubeCL proc-macro IR is huge)
[profile.dev]
debug = 0
codegen-units = 16
incremental = false

[profile.dev.build-override]
opt-level = 3
codegen-units = 256
debug = false

[profile.release]
debug = 0
incremental = false
codegen-units = 256

[profile.test]
debug = 0
codegen-units = 16
```

### 16.2 Top-level façade package — placement

Two viable placements; recommendation is option (b).

(a) **Workspace root *is* the façade** (cintx convention). The root `Cargo.toml` is *both* a workspace and a package. Pros: matches cintx exactly. Cons: top-level package mixes its own deps with workspace declaration → cintx tech-debt smell §1.4.

(b) **Workspace root is workspace-only; façade lives at `crates/pyscf-rs`** (xcfun_rs convention since it added `xcfun-rs` as a member explicitly). Recommendation: this. The root `Cargo.toml` above contains no `[package]` section. A separate `crates/pyscf-rs/Cargo.toml` re-exports.

```toml
# crates/pyscf-rs/Cargo.toml
[package]
name = "pyscf-rs"
description = "Rust quantum-chemistry library (façade)"
version.workspace = true
edition.workspace = true
license.workspace = true

[features]
default       = ["cpu"]
cpu           = ["pyscf-scf/cpu", "pyscf-dft/cpu", "pyscf-mp2/cpu", "pyscf-ccsd/cpu", "pyscf-grad/cpu"]
cuda          = ["pyscf-scf/cuda", … ]
wgpu          = ["pyscf-scf/wgpu", … ]
rocm          = ["pyscf-scf/rocm", … ]
metal         = ["wgpu"]
with-mp2      = ["dep:pyscf-mp2"]
with-ccsd     = ["dep:pyscf-ccsd"]
with-grad     = ["dep:pyscf-grad"]
with-geomopt  = ["dep:pyscf-geomopt"]

[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-runtime = { path = "../pyscf-runtime" }
pyscf-gto     = { path = "../pyscf-gto" }
pyscf-scf     = { path = "../pyscf-scf" }
pyscf-dft     = { path = "../pyscf-dft" }
pyscf-mp2     = { path = "../pyscf-mp2",     optional = true }
pyscf-ccsd    = { path = "../pyscf-ccsd",    optional = true }
pyscf-grad    = { path = "../pyscf-grad",    optional = true }
pyscf-geomopt = { path = "../pyscf-geomopt", optional = true }
```

### 16.3 Per-crate Cargo.toml summaries

Only listing distinctive elements — `version.workspace = true`, `edition.workspace = true`, `license.workspace = true` are assumed for every crate.

**`crates/pyscf-core/Cargo.toml`**
```toml
[package]
name = "pyscf-core"
description = "Core types, traits, and errors for pyscf_rs"

[dependencies]
thiserror = { workspace = true }
smallvec  = { workspace = true }
ndarray   = { workspace = true }
nalgebra  = { workspace = true }
tracing   = { workspace = true }
hdf5      = { workspace = true }   # for chkfile schema types
serde     = { workspace = true, features = ["derive"] }
```

**`crates/pyscf-runtime/Cargo.toml`** — mirrors `cintx-runtime`
```toml
[package]
name = "pyscf-runtime"
description = "Backend dispatch, planner, workspace pool"

[features]
default = ["cpu"]
cpu = []
cuda = []
wgpu = []
rocm = []
metal = []

[dependencies]
pyscf-core = { path = "../pyscf-core" }
tracing    = { workspace = true }
```

**`crates/pyscf-kernels/Cargo.toml`** — mirrors `xcfun-kernels` + `xcfun-gpu`
```toml
[package]
name = "pyscf-kernels"
description = "All cubecl kernel bodies for pyscf_rs"

[features]
default = ["cpu"]
cpu     = ["dep:cubecl-cpu",  "pyscf-runtime/cpu"]
cuda    = ["dep:cubecl-cuda", "pyscf-runtime/cuda"]
wgpu    = ["dep:cubecl-wgpu", "pyscf-runtime/wgpu"]
rocm    = ["dep:cubecl-hip",  "pyscf-runtime/rocm"]
metal   = ["wgpu",            "pyscf-runtime/metal"]
with-mp2  = []
with-ccsd = []
with-grad = []
with-dft  = []

[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-runtime = { path = "../pyscf-runtime", default-features = false }
cubecl        = { workspace = true }
cubecl-cpu    = { workspace = true, optional = true }
cubecl-cuda   = { workspace = true, optional = true }
cubecl-wgpu   = { workspace = true, optional = true }
cubecl-hip    = { workspace = true, optional = true }
bytemuck      = { workspace = true }
thiserror     = { workspace = true }
```

**`crates/pyscf-gto/Cargo.toml`**
```toml
[package]
name = "pyscf-gto"
description = "Molecular geometry and AO basis sets"

[features]
default = ["cpu"]
cpu   = ["pyscf-kernels/cpu",  "cintx/cpu"]
cuda  = ["pyscf-kernels/cuda", "cintx/cuda"]
# (other backends mirror the same shape)

[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-runtime = { path = "../pyscf-runtime", default-features = false }
pyscf-kernels = { path = "../pyscf-kernels", default-features = false }
cintx         = { workspace = true }
serde         = { workspace = true, features = ["derive"] }
serde_json    = { workspace = true }
```

**`crates/pyscf-scf/Cargo.toml`**
```toml
[package]
name = "pyscf-scf"
description = "Hartree-Fock self-consistent field methods"

[features]
default = ["cpu"]
cpu     = ["pyscf-gto/cpu", "pyscf-kernels/cpu"]
# … other backends

[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-runtime = { path = "../pyscf-runtime", default-features = false }
pyscf-kernels = { path = "../pyscf-kernels", default-features = false }
pyscf-gto     = { path = "../pyscf-gto",     default-features = false }
tracing       = { workspace = true }

[dev-dependencies]
pyscf-oracle = { path = "../pyscf-oracle" }
approx       = { workspace = true }
```

**`crates/pyscf-dft/Cargo.toml`**
```toml
[package]
name = "pyscf-dft"
description = "Density functional theory (RKS, UKS)"

[features]
default = ["cpu"]
cpu     = ["pyscf-scf/cpu", "pyscf-kernels/cpu", "pyscf-kernels/with-dft"]

[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-runtime = { path = "../pyscf-runtime", default-features = false }
pyscf-kernels = { path = "../pyscf-kernels", default-features = false }
pyscf-gto     = { path = "../pyscf-gto",     default-features = false }
pyscf-scf     = { path = "../pyscf-scf",     default-features = false }
libxc_rs      = { workspace = true }
xcfun-rs      = { workspace = true }
```

**`crates/pyscf-mp2/Cargo.toml`** — pulls `with-mp2` kernel feature
```toml
[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-kernels = { path = "../pyscf-kernels", features = ["with-mp2"] }
pyscf-gto     = { path = "../pyscf-gto" }
pyscf-scf     = { path = "../pyscf-scf" }
```

**`crates/pyscf-ccsd/Cargo.toml`** — pulls `with-ccsd` kernel feature
```toml
[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-kernels = { path = "../pyscf-kernels", features = ["with-ccsd"] }
pyscf-gto     = { path = "../pyscf-gto" }
pyscf-scf     = { path = "../pyscf-scf" }
pyscf-mp2     = { path = "../pyscf-mp2" }
```

**`crates/pyscf-grad/Cargo.toml`** — depends on every method
```toml
[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-kernels = { path = "../pyscf-kernels", features = ["with-grad"] }
pyscf-gto     = { path = "../pyscf-gto" }
pyscf-scf     = { path = "../pyscf-scf" }
pyscf-dft     = { path = "../pyscf-dft" }
pyscf-mp2     = { path = "../pyscf-mp2" }
pyscf-ccsd    = { path = "../pyscf-ccsd" }
```

**`crates/pyscf-geomopt/Cargo.toml`**
```toml
[dependencies]
pyscf-core  = { path = "../pyscf-core" }
pyscf-grad  = { path = "../pyscf-grad" }
argmin      = { workspace = true }
argmin-math = { workspace = true }
```

**`crates/pyscf-oracle/Cargo.toml`** — used as dev-dep elsewhere; this crate itself is regular
```toml
[package]
name = "pyscf-oracle"
description = "PySCF-as-live-oracle harness for parity testing"

[dependencies]
pyscf-core = { path = "../pyscf-core" }
pyo3       = { workspace = true, features = ["auto-initialize"] }
numpy      = { workspace = true }
serde      = { workspace = true, features = ["derive"] }
serde_json = { workspace = true }
approx     = { workspace = true }
anyhow     = { workspace = true }
```

**`crates/pyscf-py/Cargo.toml`** — single cdylib, mirrors `xcfun-py` exactly
```toml
[package]
name = "pyscf-py"
description = "PyO3 bindings: drop-in pyscf.* import surface"

[lib]
crate-type = ["cdylib"]

[features]
default = ["cpu"]
cpu   = ["pyscf-scf/cpu",  "pyscf-dft/cpu",  "pyscf-mp2/cpu",  "pyscf-ccsd/cpu",  "pyscf-grad/cpu"]
cuda  = ["pyscf-scf/cuda", "pyscf-dft/cuda", "pyscf-mp2/cuda", "pyscf-ccsd/cuda", "pyscf-grad/cuda"]
wgpu  = ["pyscf-scf/wgpu", … ]
rocm  = ["pyscf-scf/rocm", … ]
metal = ["wgpu"]

[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-gto     = { path = "../pyscf-gto",     default-features = false }
pyscf-scf     = { path = "../pyscf-scf",     default-features = false }
pyscf-dft     = { path = "../pyscf-dft",     default-features = false }
pyscf-mp2     = { path = "../pyscf-mp2",     default-features = false }
pyscf-ccsd    = { path = "../pyscf-ccsd",    default-features = false }
pyscf-grad    = { path = "../pyscf-grad",    default-features = false }
pyscf-geomopt = { path = "../pyscf-geomopt", default-features = false }
pyo3          = { workspace = true, features = ["extension-module", "abi3-py310"] }
numpy         = { workspace = true }
```

**`crates/pyscf-bench/Cargo.toml`**
```toml
[package]
name = "pyscf-bench"
publish = false

[dependencies]
pyscf-core = { path = "../pyscf-core" }
pyscf-scf  = { path = "../pyscf-scf" }
pyscf-dft  = { path = "../pyscf-dft" }
pyscf-mp2  = { path = "../pyscf-mp2" }
pyscf-ccsd = { path = "../pyscf-ccsd" }
pyscf-grad = { path = "../pyscf-grad" }

[dev-dependencies]
criterion = { workspace = true }
```

---

## 17. Sources

- `/home/user/Documents/workspace/cintx/Cargo.toml` (workspace root + `cintx` package)
- `/home/user/Documents/workspace/cintx/crates/cintx-core/Cargo.toml`
- `/home/user/Documents/workspace/cintx/crates/cintx-ops/Cargo.toml`
- `/home/user/Documents/workspace/cintx/crates/cintx-runtime/Cargo.toml`
- `/home/user/Documents/workspace/cintx/crates/cintx-cubecl/Cargo.toml`
- `/home/user/Documents/workspace/cintx/crates/cintx-compat/Cargo.toml`
- `/home/user/Documents/workspace/cintx/crates/cintx-rs/Cargo.toml`
- `/home/user/Documents/workspace/cintx/crates/cintx-capi/Cargo.toml`
- `/home/user/Documents/workspace/cintx/crates/cintx-oracle/Cargo.toml`
- `/home/user/Documents/workspace/cintx/README.md` (source-tree section)
- `/home/user/Documents/workspace/xcfun_rs/Cargo.toml`
- `/home/user/Documents/workspace/xcfun_rs/crates/xcfun-{core,ad,kernels,eval,gpu,rs,capi,py}/Cargo.toml`
- `/home/user/Documents/workspace/xcfun_rs/crates/xcfun-py/src/lib.rs` (PyO3 0.28 module pattern)
- `/home/user/Documents/workspace/libxc_rs/Cargo.toml` (counter-example: flat-kernels pattern, NOT followed)
- `/home/user/Documents/workspace/pyscf_rs/.planning/PROJECT.md` (constraints + sister-project naming)
- `/home/user/Documents/workspace/pyscf_rs/.planning/codebase/{ARCHITECTURE,STRUCTURE,INTEGRATIONS}.md` (upstream PySCF module map and env vars)

Confidence: **HIGH** for the sibling-pattern observations (read directly from `Cargo.toml`s), **HIGH** for the DAG and crate split (logically derived from PySCF's known module dependencies), **MEDIUM** for the specific feature-flag matrix on `pyscf-kernels` (a couple of design choices — e.g. `with-mp2` vs always-on — could legitimately be settled either way; explicit feature gates were chosen to keep SCF-only build size sane).

---
*Architecture research for: pyscf_rs (Rust quantum-chemistry library, cubecl + PyO3)*
*Researched: 2026-05-09*

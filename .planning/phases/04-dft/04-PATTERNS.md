# Phase 4: DFT - Pattern Map

**Mapped:** 2026-05-22
**Files analyzed:** 24 new/modified files
**Analogs found:** 22 with analog / 24 total (2 ports have no Rust analog — port the upstream Python algorithm)

> **Read-only sibling crates:** `~/Documents/workspace/{libxc_rs,xcfun_rs,cintx}` are READ-ONLY. Never run `cargo build/test/check --features libxc` locally (266 kernel crates, ~6h freeze). All sibling-crate excerpts below are from source-read only.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/pyscf-grids/Cargo.toml` | config | — | `crates/pyscf-df/Cargo.toml` | exact (algebra-dep split-out crate) |
| `crates/pyscf-grids/src/lib.rs` | lib root | — | `crates/pyscf-df/src/lib.rs` | exact |
| `crates/pyscf-grids/src/lebedev.rs` | utility (generator) | transform | `crates/pyscf-kernels/src/eval_gto.rs` (algorithm port) | partial (port; no Rust analog) |
| `crates/pyscf-grids/src/radial.rs` | utility (formula) | transform | `crates/pyscf-df/src/cholesky_eri.rs` (algebra-orchestrated port) | role-match |
| `crates/pyscf-grids/src/radii.rs` | data tables | — | `crates/pyscf-df/src/auxbasis.rs` (const tables) | role-match |
| `crates/pyscf-grids/src/partition.rs` | service | transform/reduce | `crates/pyscf-df/src/df_jk.rs` (oracle_sum reductions) | role-match |
| `crates/pyscf-grids/src/prune.rs` | utility | transform | `crates/pyscf-df/src/auxbasis.rs` | partial |
| `crates/pyscf-grids/src/levels.rs` | data tables | — | `crates/pyscf-df/src/auxbasis.rs` (`DEFAULT_AUXBASIS`) | role-match |
| `crates/pyscf-dft/src/lib.rs` | lib root | — | `crates/pyscf-scf/src/lib.rs` | exact |
| `crates/pyscf-dft/src/rks.rs` (RKS struct) | model/driver | request-response | `crates/pyscf-scf/src/rhf.rs` (RHF + kernel reuse) | exact |
| `crates/pyscf-dft/src/uks.rs` (UKS struct) | model/driver | request-response | `crates/pyscf-scf/src/uhf.rs` | exact |
| `crates/pyscf-dft/src/hooks.rs` (`KsOverrideHooks`) | trait/middleware | request-response | `crates/pyscf-scf/src/hooks.rs` (`OverrideHooks`) | exact |
| `crates/pyscf-dft/src/veff.rs` (`get_veff`) | service | CRUD/transform | `crates/pyscf-scf/src/fock.rs` (`default_get_veff`/`get_jk`) | exact |
| `crates/pyscf-dft/src/numint.rs` (`NumInt`) | service | streaming (grid block-loop) | `crates/pyscf-df/src/df_jk.rs` (block + oracle_sum) | role-match |
| `crates/pyscf-dft/src/xc_backend.rs` (`XcBackend`) | service (dispatch enum) | request-response | `crates/pyscf-algebra/src/client.rs` (`AlgebraClient` cfg-match) | exact |
| `crates/pyscf-dft/src/parser/libxc.rs` (`parse_xc`) | utility (parser) | transform | `crates/pyscf-gto` algorithm-port precedent (`format_atom`/`make_env`) | partial (port; no Rust analog) |
| `crates/pyscf-dft/src/parser/xcfun.rs` (`parse_xc`) | utility (parser) | transform | `crates/pyscf-dft/src/parser/libxc.rs` (same shape) | role-match |
| `crates/pyscf-dft/src/vv10.rs` (NLC) | service | streaming (double-loop) | `crates/pyscf-df/src/df_jk.rs` (nested oracle_sum) | role-match |
| `crates/pyscf-dft/src/chkfile.rs` (`KsResult`) | persistence | file-I/O | `crates/pyscf-scf/src/chkfile.rs` (`impl Checkpointable for ScfResult`) | exact |
| `crates/pyscf-dft/src/df_dft.rs` (`density_fit`) | service | CRUD | `crates/pyscf-scf/src/df_scf.rs` (`RHF::density_fit` + `DfHooks`) | exact |
| `crates/pyscf-dft/Cargo.toml` | config | — | `crates/pyscf-df/Cargo.toml` + `xcfun-rs` `[features]` block | exact + role-match |
| `crates/pyscf-py/src/dft.rs` (PyRKS/PyUKS) | binding/controller | request-response | `crates/pyscf-py/src/scf.rs` (PyRHF/PyUHF) | exact |
| `crates/pyscf-py/src/bridge.rs` (`KsOverrideHooks` impl) | binding/middleware | request-response | `crates/pyscf-py/src/bridge.rs` (`PyOverrideBridge`) | exact (extend existing) |
| `python/pyscf/dft/__init__.py` | overlay | — | `python/pyscf/scf/__init__.py` | exact |
| `Cargo.toml` (workspace) | config | — | `Cargo.toml:18-24` (Phase 3 18-member additions) | exact |
| `xtask/src/bin/check_dependency_wall.rs` | config (lint) | — | (NO EDIT — see Shared Patterns / Pitfall) | n/a |

---

## Pattern Assignments

### `crates/pyscf-grids/Cargo.toml` (config — new 19th member)

**Analog:** `crates/pyscf-df/Cargo.toml` (algebra-dep split-out crate, lines 1-18)

The new grids crate is structurally identical to `pyscf-df`/`pyscf-diis` — a Phase-3-style cross-method-shared crate that depends on `pyscf-core` + `pyscf-algebra` (the algebra wall keeps it OUT of the cubecl carve-out). Copy this skeleton verbatim, dropping `pyscf-gto` if grids only needs `Mole` geometry (it does need atom coords + Z, available on `pyscf-core::Mole`):

```toml
# from crates/pyscf-df/Cargo.toml — the canonical algebra-dep split-out shape
[package]
name = "pyscf-grids"
version.workspace      = true
edition.workspace      = true
rust-version.workspace = true
license.workspace      = true
description = "Becke molecular grids (gen_grid/radi/Lebedev port). Phase 4 (D-05)."

[lib]
path = "src/lib.rs"

[dependencies]
pyscf-core    = { path = "../pyscf-core" }
pyscf-algebra = { path = "../pyscf-algebra" }   # Becke-weight ordered reductions
thiserror     = { workspace = true }
tracing       = { workspace = true }
```

`pyscf-diis/Cargo.toml` (lines 12-15) is the even-tighter `pyscf-core + pyscf-algebra + thiserror` variant if `tracing` isn't needed. Note: NO `cubecl-*` dep — the wall (see Shared Patterns) forbids it; route every reduction through `pyscf-algebra::oracle_sum`.

---

### `crates/pyscf-grids/src/lib.rs` (lib root)

**Analog:** `crates/pyscf-df/src/lib.rs` (lines 1-25)

Copy the module-doc-comment + `#![forbid(unsafe_code)]` + `#![warn(clippy::unwrap_used)]` + `pub mod`/`pub use` shape exactly:

```rust
//! pyscf-grids: Becke molecular grids.
//!
//! Source: D-05/D-06 — Phase 4 introduces this crate. Mirrors upstream
//! `pyscf/dft/{gen_grid.py, radi.py, LebedevGrid.py}` + `pyscf/data/radii.py`.
#![forbid(unsafe_code)]
#![warn(clippy::unwrap_used)]

pub mod radial;
pub mod radii;
pub mod lebedev;
pub mod prune;
pub mod partition;
pub mod levels;

pub use ...; // re-export Grids, gen_atomic_grids, etc.
```

The lib-root crate-doc-comment convention: name the upstream Python source files being ported and the deciding D-number — see `pyscf-df/src/lib.rs:1-13` for the exemplar (it names `df.py/df_jk.py/incore.py/addons.py` and D-10).

---

### `crates/pyscf-grids/src/lebedev.rs` + `radial.rs` + `radii.rs` + `prune.rs` + `levels.rs` (algorithm + const-table ports)

**Analog:** `crates/pyscf-kernels/src/eval_gto.rs` (port-the-reference-algorithm precedent) — there is **no existing Rust analog** for the Lebedev generator; this is a fresh port. The *style* analog is `eval_gto.rs`.

The established port convention (from `eval_gto.rs:1-12` doc header):
1. Module doc names the upstream `.py` source, line range, license, and the algorithm in prose.
2. The const seed tables (`LEBEDEV_ORDER`, `BRAGG`, `COVALENT`, `SG1RADII`) are **inline `const` arrays**, NOT generated, NOT `build.rs`, NOT runtime file reads (D-06 + "don't freeze compile" user memory). Pattern for const data tables: `crates/pyscf-df/src/auxbasis.rs` `DEFAULT_AUXBASIS` (a `&[(&str, &str)]` const slice with a lookup fn).
3. Bit-exactness rides on `release-oracle` (FMA-free) + ordered reductions — any sum of >2 terms in the radial/angular product uses `pyscf_algebra::oracle_sum` (Pitfall 10, this phase's owned pitfall).

`eval_gto.rs:1-12` header to mirror:
```rust
//! GTO-07 + D-04: AO-on-grid kernel (`eval_gto`).
//!
//! Source: pyscf/gto/eval_gto.py (Apache-2.0). The reference algorithm:
//! per grid point g, walk every shell s, compute the contracted radial ...
```

**Critical class-vs-function defaults** (RESEARCH Pitfall 3, gen_grid.py:490-499): use the `Grids` *class* defaults — `radi_method=radi.treutler`, `radii_adjust=treutler_atomic_radii_adjust`, `becke_scheme=original_becke`, `prune=nwchem_prune`, `atomic_radii=BRAGG_RADII`, `level=3`. NOT the `gen_atomic_grids` function defaults.

---

### `crates/pyscf-grids/src/partition.rs` (Becke partition — service, reduce)

**Analog:** `crates/pyscf-df/src/df_jk.rs` (lines 34-117 — the `oracle_sum`-per-output-element reduction pattern)

The Becke partition normalization (`pbecke.sum(axis=0)`, gen_grid.py:337) is the byte-exactness hot spot (Pitfall 10). Port the **pure-Python `get_partition` fallback** (gen_grid.py:314-329), NOT the C kernel `libdft.VXCgen_grid` (RESEARCH Pitfall 4 — the C kernel is PySCF's own `libdft`, not in any sibling crate). Mirror the `df_jk.rs` reduction discipline: build a per-element terms buffer, then `oracle_sum`:

```rust
// from crates/pyscf-df/src/df_jk.rs:73-84 — the ordered-reduction convention
rho_q[q] = pyscf_algebra::oracle_sum(&prod_buf);
...
j[mu * nao + nu] = pyscf_algebra::oracle_sum(&row_buf);
```

`pyscf-diis/src/lib.rs:6-12` documents the same Pitfall-9 discipline ("all reductions go through `pyscf-algebra::oracle_dot`/`oracle_sum`"). Apply identically to the partition-weight sum.

---

### `crates/pyscf-dft/src/lib.rs` (lib root — fills the Phase 1 stub)

**Analog:** `crates/pyscf-scf/src/lib.rs` (lines 1-66)

Current state is a 5-line empty stub (`crates/pyscf-dft/src/lib.rs:1-5`). Replace with the `pyscf-scf` lib-root shape: crate-doc naming the upstream sources + the plan that fills it, `#![forbid(unsafe_code)]` + `#![warn(clippy::unwrap_used)]`, then `pub mod` declarations and grouped `pub use` re-exports. The `pyscf-scf` re-export grouping (lines 37-65) is the model — re-export the trait + structs + the `default_*` free fns:

```rust
// shape from crates/pyscf-scf/src/lib.rs:37-43
pub use error::DftError;          // (or reuse PyscfRsError + a Dft variant)
pub use hooks::{KsOverrideHooks, NoKsOverrides};
pub use rks::RKS;
pub use uks::UKS;
pub use numint::NumInt;
pub use xc_backend::XcBackend;
```

---

### `crates/pyscf-dft/src/rks.rs` + `uks.rs` (RKS/UKS struct + kernel reuse — model/driver)

**Analog:** `crates/pyscf-scf/src/rhf.rs` (lines 23-178) — the RKS struct mirrors RHF *plus* DFT attributes (`xc`, `grids`, `nlc`, `nlcgrids`, `_numint`).

The headline pattern: **RKS reuses the Phase 3 generic `kernel<H>` wholesale**, swapping in a KS hooks impl that overrides `get_veff`. Copy the `RHF::kernel` body (rhf.rs:138-148) — it builds a `KernelConfig`, calls the generic `kernel(&mol, &hooks, cfg)`, and copies the result back into struct state:

```rust
// from crates/pyscf-scf/src/rhf.rs:138-148 — RKS::kernel reuses this verbatim,
// passing KS hooks instead of NoOverrides
pub fn kernel(&mut self) -> Result<ScfResult, PyscfRsError> {
    let cfg = self.to_kernel_config();
    let result = kernel(&self.mol, &self.ks_hooks(), cfg)?;   // KS get_veff override
    self.mo_coeff = Some(result.mo_coeff.clone());
    self.e_tot = result.e_tot.0;
    self.converged = result.converged;
    self.cycles = result.cycles;
    Ok(result)
}
```

Also copy the `to_kernel_config` field-mapping (rhf.rs:153-177) and the manual `Debug` impl pattern (rhf.rs:58-95) for any opaque slots (e.g. `_numint`, `grids`). The 30-attribute floor convention (rhf.rs doc-header:4-11) extends with DFT-specific fields.

`uks.rs` analog is `crates/pyscf-scf/src/uhf.rs` (the alpha/beta open-shell variant).

---

### `crates/pyscf-dft/src/hooks.rs` (`KsOverrideHooks` trait — middleware)

**Analog:** `crates/pyscf-scf/src/hooks.rs` (lines 13-131)

`KsOverrideHooks` extends the Phase 3 `OverrideHooks` shape. The trait + a `NoOverrides`-style default-impl struct is the exact pattern. Copy the trait-method signature style (each returns `Result<T, PyscfRsError>`, takes `&self`) and the `NoOverrides` impl that delegates each method to a `default_*` free fn:

```rust
// from crates/pyscf-scf/src/hooks.rs:13-26 — KsOverrideHooks adds get_veff (DFT
// form: J + Vxc - hyb*K), define_xc_, and inherits the SCF hooks
pub trait OverrideHooks {
    fn get_hcore(&self, mol: &Mole) -> Result<Density, PyscfRsError>;
    ...
    fn get_veff(&self, mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError>;
    ...
}
```

`NoOverrides` delegation pattern (hooks.rs:63-131): each method body is `crate::<module>::default_<hook>(args)`. The doc-comment honesty convention (hooks.rs:1-10) — explain the logical-count-vs-surface-count discrepancy — should be replicated for the DFT hook expansion.

The `pyscf-dft` crate stays **pyo3-free** (D-01); the `KsOverrideHooks` PyO3 impl lives in `pyscf-py` (see bridge entry). This is the algebra-wall analog for PyO3.

---

### `crates/pyscf-dft/src/veff.rs` (`get_veff` — service, transform)

**Analog:** `crates/pyscf-scf/src/fock.rs` (lines 79-148 — `default_get_jk` + `default_get_veff`)

KS `get_veff = J + Vxc − (hyb/RSH-scaled) K`. The exact-exchange K reuses the Phase 3 `get_jk` path; the only new compute is `+ Vxc` (from `NumInt.nr_rks`) and the `hyb` scaling of K. Copy the `default_get_veff` structure (fock.rs:140-148) which fetches `(j, k)` then forms the linear combination element-wise (note: inline loop, not `axpy` — `pyscf_algebra::axpy` is Tensor-API/`NotYetImplemented{phase:2}`, see fock.rs comment + df_scf.rs:108-115):

```rust
// from crates/pyscf-scf/src/fock.rs:140-148 — KS form scales K by hyb and adds Vxc
pub fn default_get_veff(mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError> {
    let (j, k) = default_get_jk(mol, dm)?;
    let nao = j.nao;
    let mut data = vec![0.0_f64; nao * nao];
    for i in 0..(nao * nao) {
        data[i] = j.data[i] - 0.5 * k.data[i];   // KS: j[i] + vxc[i] - hyb*k[i]
    }
    Ok(Density { nao, data })
}
```

**RSH branch (DFT-05):** RESEARCH Pitfall 1 — do NOT use `int2e_lr_*`/`int2e_sr_*` symbols (they don't exist in cintx). Set `PTR_RANGE_OMEGA = env[8]` around the standard `int2e` call on the `pyscf-gto::intor` path (see intor.rs:69 dispatcher; the env-slot setter is a small `pyscf-gto` addition — Open Question A5/Q1, planner must scope). Mirror upstream rks.py:108-129.

---

### `crates/pyscf-dft/src/numint.rs` (`NumInt` grid block-loop — service, streaming)

**Analog:** `crates/pyscf-df/src/df_jk.rs` (lines 34-117 — the block + per-element `oracle_sum` reduction chain, plus its module-doc Pitfall-9 statement)

The `nr_rks`/`nr_uks` grid loop (D-07) is orchestrated through `pyscf-algebra` — no bespoke cubecl kernel. The per-block chain: AO = `eval_gto` (`pyscf-kernels`), ρ = AOᵀ·D·AO via `pyscf_algebra::gemm`, XC via `XcBackend.eval`, Vxc back-contraction via weighted `gemm`, Exc += `oracle_sum(w·exc·ρ)`. The `df_jk.rs` reduction discipline is the closest existing analog for the bit-exact accumulation:

```rust
// from crates/pyscf-df/src/df_jk.rs:73,84 — every accumulation is oracle_sum
rho_q[q]            = pyscf_algebra::oracle_sum(&prod_buf);
j[mu * nao + nu]    = pyscf_algebra::oracle_sum(&row_buf);
```

`gemm` call sites: `pyscf_algebra::gemm` (re-exported `crates/pyscf-algebra/src/lib.rs:50`); authoritative call convention in `docs/manual/Cubecl/cubecl_matmul_gemm_example.md`. Signature contract (DFT-10): `nr_rks(ni, mol, grids, xc_code, dms, relativity, hermi, max_memory, verbose) -> (nelec, excsum, vmat)` — port from numint.py:1074. blksize chunking: mirror upstream MEM-driven blksize, log `PYSCF_MAX_MEMORY` at entry (Phase 3 SCF convention), no hard enforcement in v1.

---

### `crates/pyscf-dft/src/xc_backend.rs` (`XcBackend` dispatch enum — service)

**Analog:** `crates/pyscf-algebra/src/client.rs` (lines 10-39 — `AlgebraClient` enum + `cfg`-gated match dispatch)

The `XcBackend` seam (D-01/D-03) is structurally the `AlgebraClient` enum: a variant per backend, each non-default variant `#[cfg(feature = ...)]`-gated, with a `match self` dispatch. The default build must never name `libxc_rs` symbols:

```rust
// shape from crates/pyscf-algebra/src/client.rs:10-37 — AlgebraClient's cfg-gated
// enum + match dispatch is the exact model for XcBackend
pub enum AlgebraClient {
    Cpu(ComputeClient<CpuRuntime>),
    #[cfg(feature = "cuda")] Cuda(...),
    #[cfg(feature = "wgpu")] Wgpu(...),
}
impl AlgebraClient {
    pub fn kind(&self) -> BackendKind {
        match self {
            Self::Cpu(_) => BackendKind::Cpu,
            #[cfg(feature = "cuda")] Self::Cuda(_) => BackendKind::Cuda,
            ...
        }
    }
}
```

For `XcBackend`: `Xcfun` is default-compiled; `#[cfg(feature = "libxc")] Libxc`; the `#[cfg(not(feature = "libxc"))]` arm returns `PyscfRsError::NotYetImplemented` / a `LibxcFeatureNotEnabled` error (RESEARCH Pattern 1, lines 222-240). The Cargo wiring (`libxc = ["dep:libxc_rs"]`, `libxc_rs = { path = "...", optional = true }`) follows `pyscf-kernels/Cargo.toml`'s per-backend optional-dep pattern (lines `[features]` block) AND the `xcfun-rs` `[features]` block (the canonical model for what D-04 must add to `libxc_rs`).

---

### `crates/pyscf-dft/src/parser/libxc.rs` + `xcfun.rs` (`parse_xc` — utility, transform)

**Analog:** No Rust analog. Port-the-reference-algorithm precedent is the Phase 2 `format_atom`/`make_env` ports + `eval_gto.rs` style. The two parsers share a shape; `xcfun.rs` mirrors `libxc.rs`.

Port `pyscf/dft/libxc.py:parse_xc` (lines 491-718) + `XC_CODES` (154) + `XC_ALIAS` (217) as the **default** resolver routing to `libxc_rs`; `pyscf/dft/xcfun.py:parse_xc` (lines ~) as the alternate routing to `xcfun_rs` (D-01). Return shape `(hyb, alpha, omega), ((libxc_id, fac), ...)`. The `XC_CODES`/`XC_ALIAS` tables follow the same inline-`const`-table convention as `auxbasis.rs` `DEFAULT_AUXBASIS` (NO codegen, NO build.rs). Token sub-cases the parser must handle (RESEARCH Code Examples §1): `-` sign prefix, `*` factor either order, `E_`→`E-` exponent fixup, `RSH(alpha;beta;omega)`, `HF`, `SR_HF`/`LR_HF`, raw integer ID, comma X/C split, compound-name expansion, `0.5*b3lyp` unit-scaling.

**libxc_rs eval path (RESEARCH Pattern 2, verified API):** `libxc_rs::lookup_by_name(name) -> FunctionalId` (registry/mod.rs:40), `lookup_by_id(u16) -> &FunctionalMeta` (registry/mod.rs:12), `FunctionalBuilder::new(id)` (builder.rs:53), `BatchEvaluator::new(spin, np_max)` (batch.rs:43) + `.evaluate(&func, &input, DerivativeOrder, &mut out)` (batch.rs:61). Family dispatch via `FunctionalMeta.family()/.kind()` sizes `LdaInput`/`GgaInput`/`MggaInput`. (A3: re-confirm exact constructor/field names against `src/api/batch.rs` + `src/output/*` when writing — read only, never compile.)

**xcfun_rs eval path (RESEARCH Pattern 3, verified):** `Functional::new()` (functional.rs:164), `.set(name, weight)` (189), `.is_gga()`/`.is_metagga()` (202/208), `.eval_setup(Vars, Mode, order)` (221), `.input_length()`/`.output_length()` (271/296), `.eval_vec(...)` (352). (A4: confirm `Vars`/`Mode` variant spellings against `xcfun-core/src/enums.rs`.)

---

### `crates/pyscf-dft/src/vv10.rs` (VV10 NLC — service, double-loop)

**Analog:** `crates/pyscf-df/src/df_jk.rs` (nested-loop + `oracle_sum` inner reductions)

Port the **pure-Python `_vv10nlc` double-loop** (numint.py:526-538 commented block), NOT the C kernel `libdft.VXC_vv10nlc` (Pitfall 4). Embarrassingly parallel over outer grid points; use `oracle_sum` for the inner-grid reductions (bit-exact). The `df_jk.rs` nested-reduction pattern (build inner terms buffer → `oracle_sum`) applies directly. Constants/coeffs: query `libxc_rs::NlcCoefficients` per-functional (A1 — don't hardcode except the bare `'VV10'` default b=5.9/C=0.0093). Uses a second coarser `pyscf-grids` instance (`nlcgrids`).

---

### `crates/pyscf-dft/src/chkfile.rs` (`impl Checkpointable for KsResult` — persistence, file-I/O)

**Analog:** `crates/pyscf-scf/src/chkfile.rs` (lines 21-123 — `impl Checkpointable for ScfResult` + `dump_*_to_file`/`load_*_from_file`)

This is an exact-shape copy. The `Checkpointable` trait (`pyscf-chkfile/src/checkpointable.rs:14-20`) is `dump(&self, group) + load(group)`. Copy the SCF impl structure (chkfile.rs:21-97): write scalar f64 + 1D datasets + the F-order `mo_coeff` via `write_dataset_f_order`, then add the DFT `xc`/`grids` metadata to the schema. Use `pyscf_chkfile::primitives::*` (re-exported via the crate's hdf5 alias — chkfile/src/lib.rs:33-37 keeps `pyscf-dft` from adding its own `hdf5-metno` dep, the D-05 sole-owner discipline):

```rust
// from crates/pyscf-scf/src/chkfile.rs:21-25 — KsResult mirrors this; add xc/grids meta
impl Checkpointable for ScfResult {
    fn dump(&self, scf_group: &hdf5::Group) -> Result<(), ChkfileError> {
        primitives::write_scalar_f64(scf_group, "e_tot", self.e_tot.0)?;
        primitives::write_dataset_1d(scf_group, "mo_energy", &self.mo_energy)?;
        ...
```

The doc-header convention (chkfile.rs:1-14) names the upstream `pyscf/*/chkfile.py` source + Pitfall 8 (F-order mo_coeff). `checkpointable.rs:5-6` explicitly anticipates `pyscf-dft::KsResult (Phase 4)`.

---

### `crates/pyscf-dft/src/df_dft.rs` (`density_fit` — service, CRUD)

**Analog:** `crates/pyscf-scf/src/df_scf.rs` (lines 26-163 — `RHF::density_fit` + `DfHooks`)

DFT-07 reuses `pyscf-df::DfIntegrals`/`get_jk_df` for the Coulomb-J build (D-10, no new crate). Copy the two-part pattern: (1) `RKS::density_fit(auxbasis)` pre-computes B integrals via `pyscf_df::cholesky_eri` and stores in `with_df` as `Box<dyn Any + Send + Sync>` (df_scf.rs:42-54); (2) a `DfKsHooks` `KsOverrideHooks` impl that routes `get_jk`/the J-part-of-`get_veff` through `pyscf_df::get_jk_df` while delegating other hooks to `default_*` (df_scf.rs:79-163):

```rust
// from crates/pyscf-scf/src/df_scf.rs:42-54 — RKS::density_fit mirrors this shape
pub fn density_fit(mut self, auxbasis: Option<&str>) -> Result<Self, PyscfRsError> {
    let basis_name_owned = extract_basis_name(&self.mol.basis);
    let aux = auxbasis.unwrap_or_else(|| default_jkfit(&basis_name_owned));
    let df = cholesky_eri(&self.mol, aux)?;
    self.with_df = Some(Box::new(df));
    Ok(self)
}
```

The `DfHooks` get_jk override (df_scf.rs:96-98) is the exact J-build seam; for DFT the Vxc/K parts stay on the standard grid-loop/get_jk path.

---

### `crates/pyscf-dft/Cargo.toml` (config)

**Analog:** `crates/pyscf-df/Cargo.toml` (dep list) + `crates/pyscf-kernels/Cargo.toml` `[features]` block + `xcfun-rs/Cargo.toml` `[features]` block (the optional-dep / off-by-default model)

Current state is an empty `[dependencies]` (verified). Fill with: `pyscf-core`, `pyscf-algebra`, `pyscf-gto`, `pyscf-scf`, `pyscf-df`, `pyscf-grids` (new), `pyscf-chkfile`, `pyscf-runtime`, `xcfun-rs`, `tracing`, `thiserror`. NO pyo3 dep (algebra/pyo3 wall). `libxc_rs` is **optional behind `features = ["libxc"]`** — the off-by-default umbrella-feature pattern from `pyscf-kernels/Cargo.toml` (`[features] default = ["cpu"]` + `cuda = ["dep:cubecl-cuda"]`):

```toml
# off-by-default optional dep — model: pyscf-kernels/Cargo.toml [features] +
# xcfun-rs/Cargo.toml. Pitfall 5: the dep MUST be optional or every build
# pulls libxc_rs's 266 kernels (~6h freeze).
[features]
default = []
libxc   = ["dep:libxc_rs"]

[dependencies]
libxc_rs = { path = "../../../libxc_rs", optional = true }
```

The `xcfun-rs` `[features]` block (read this session, lines ~9-26) is the canonical model for the D-04 cross-crate `libxc_rs` `[features]` gap-closure (make the 266 kernel deps `optional = true` + per-functional features + `cfg`-gated dispatch — RESEARCH Runtime State Inventory).

---

### `crates/pyscf-py/src/dft.rs` (PyRKS/PyUKS + `dft` submodule — binding)

**Analog:** `crates/pyscf-py/src/scf.rs` (lines 1-160) + `crates/pyscf-py/src/lib.rs` (lines 40-57, the `_native` pymodule + submodule registration)

PyRKS/PyUKS mirror PyRHF/PyUHF exactly: `#[pyclass(subclass, name = "RKS", module = "pyscf._native.dft")]` wrapping an inner `RksRust` + `py_mol: Py<PyAny>`, with `#[getter]/#[setter]` pairs for the attribute floor and a `register(py, m)` fn. Copy the pyclass + register shape:

```rust
// from crates/pyscf-py/src/scf.rs:31-61 — PyRKS mirrors this exactly
pub(crate) fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRKS>()?;
    m.add_class::<PyUKS>()?;
    Ok(())
}

#[pyclass(subclass, name = "RKS", module = "pyscf._native.dft")]
pub struct PyRKS {
    pub(crate) inner: RksRust,
    pub(crate) py_mol: Py<PyAny>,
}
```

Submodule wiring in `lib.rs` (mirror scf.rs registration at lib.rs:51-54):
```rust
// add to crates/pyscf-py/src/lib.rs _native() — alongside the existing scf submodule
let dft_mod = PyModule::new(py, "dft")?;
crate::dft::register(py, &dft_mod)?;
m.add_submodule(&dft_mod)?;
```

Also wire the Phase 3 `to_uks`/`to_rks` stubs (convert.rs:82-95 currently return `NotYetImplemented{phase:4}`) to real KS targets.

---

### `crates/pyscf-py/src/bridge.rs` (`KsOverrideHooks` impl — extend existing)

**Analog:** `crates/pyscf-py/src/bridge.rs` (lines 33-291 — `PyOverrideBridge` + `impl OverrideHooks`)

Extend (don't replace) the existing `PyOverrideBridge`. The DFT hooks (`get_veff` DFT-form, `define_xc_` string form) get the same `Python::attach` + `call_hook(&self.slf, "hook_name", args, extract)` dispatch (bridge.rs:55-70 helper). Each hook packages Rust args into NumPy arrays via `density_to_pyarray`, calls `slf.call_method1`, converts the result back. Copy the `get_veff` impl (bridge.rs:150-163) as the template for the KS `get_veff` override:

```rust
// from crates/pyscf-py/src/bridge.rs:150-163 — KS get_veff override reuses this
fn get_veff(&self, _mol: &Mole, dm: &Density) -> Result<Density, PyscfRsError> {
    Python::attach(|py| {
        let dm_py = density_to_pyarray(py, dm).map_err(py_to_pyscf)?;
        let args = PyTuple::new(py, [self.py_mol.bind(py).clone(), dm_py.into_any()])
            .map_err(py_to_pyscf)?;
        call_hook(&self.slf, "get_veff", args, |r| {
            let arr: numpy::PyReadonlyArray2<f64> = r.extract()?;
            to_density(arr)
        })
    })
}
```

`define_xc_` string form (D-02): just feeds another XC string to the D-01 parser; the Python-callable form is stubbed `NotYetImplemented{deferred}` (it would break the `Python::detach` seam — bridge doc-comment honesty convention applies).

---

### `python/pyscf/dft/__init__.py` (overlay)

**Analog:** `python/pyscf/scf/__init__.py` (lines 1-9)

Exact-shape copy — re-export the native classes from the submodule:

```python
# mirror python/pyscf/scf/__init__.py exactly
"""pyscf.dft overlay — re-exports from pyscf._native.dft (BIND-02 analog)."""
from pyscf._native.dft import RKS, UKS  # type: ignore[attr-defined]

__all__ = ["RKS", "UKS"]
```

Also add `from pyscf._native import dft` to `python/pyscf/__init__.py` (mirror its `from pyscf._native import scf` at line 9) so `from pyscf import dft` resolves to the overlay.

---

### `Cargo.toml` (workspace — add member, re-enable patch behind feature)

**Analog:** `Cargo.toml:18-24` (the Phase 3 18-member additions block) + `Cargo.toml:107-110` (the `[patch.crates-io]` block with the commented `libxc_rs` line)

Add `"crates/pyscf-grids"` to `members` (18→19) with a comment matching the Phase 3 additions style (lines 18-19). Re-enable the `libxc_rs` `[patch.crates-io]` entry (line 109 is already commented with the exact re-enable note) — **behind the feature** (Pitfall 5: the patch entry can be uncommented, but nothing references `libxc_rs` unless `--features libxc` is on, and the `pyscf-dft` dep must be `optional = true`):

```toml
# Cargo.toml:109 — the entry to re-enable; keep the comment honest
# libxc_rs  = { path = "../libxc_rs" }   # disabled — pulls a ~6h libxc compile; re-enable when Phase 4 (DFT) needs it
```

ROADMAP.md + STATE.md `total members` need the 18→19 update (D-05, RESEARCH Runtime State Inventory).

---

## Shared Patterns

### Algebra wall (cubecl containment) — DO NOT EDIT THE LINT

**Source:** `xtask/src/bin/check_dependency_wall.rs` (lines 28-47)
**Apply to:** `pyscf-grids`, `pyscf-dft` (both stay subject to the wall)

The lint is a **denylist with a carve-out**, NOT an allowlist. `FORBIDDEN_DEPS` (lines 28-38) lists every `cubecl-*` crate; `ALLOWED_CRATES = ["pyscf-algebra", "pyscf-runtime", "pyscf-kernels"]` (line 47) is the only carve-out. Any `pyscf-*` crate NOT in the carve-out fails the lint if it names a cubecl dep.

**CRITICAL (RESEARCH Pitfall 2):** The CONTEXT integration note "extend allowlist for pyscf-grids + pyscf-dft" is INVERTED. `pyscf-grids` and `pyscf-dft` route through `pyscf-algebra` and MUST stay OUT of `ALLOWED_CRATES`. Adding them would defeat the wall. **The correct edit to this file is NONE.** Reject any PR diff that adds `pyscf-grids`/`pyscf-dft` to `ALLOWED_CRATES`.

### Bit-exact ordered reductions

**Source:** `crates/pyscf-algebra/src/lib.rs:53` (`oracle_sum`/`oracle_dot`/`oracle_einsum`); usage exemplar `crates/pyscf-df/src/df_jk.rs:73-117`
**Apply to:** every reduction in `pyscf-grids` (Becke partition normalization), `numint.rs` (Exc accumulation, ρ-contraction), `vv10.rs` (inner double-loop)

Never use naive `iter().sum()`. The `pyscf-diis`/`pyscf-df` crates document this as Pitfall-9 mitigation; Phase 4 owns Pitfall 10 (grid-weight byte-exactness) which rides entirely on this + the `release-oracle` FMA-free profile (`Cargo.toml:88-94`).

### Error type + `NotYetImplemented` deferral marker

**Source:** `crates/pyscf-core/src/error.rs:15-43` (`PyscfRsError` + `NotYetImplemented { phase, what }`)
**Apply to:** all `pyscf-dft`/`pyscf-grids` fallible functions; the deferred `define_xc_` callable form; the `libxc`-feature-off path

Reuse `PyscfRsError` (or add a `#[from]` `DftError` variant following the `BasisLoad`/`EcpLoad` pattern at error.rs:30-42). The `NotYetImplemented { phase, what }` marker is the deferral convention (convert.rs:82-95 used `phase: 4` for the to_uks/to_rks stubs you now wire; use `phase: 8` for fused-kernel deferrals, a `deferred` marker for D-02's callable XC).

### Module doc-comment provenance convention

**Source:** every Phase 1-3 module (e.g. `pyscf-scf/src/fock.rs:1-27`, `pyscf-df/src/lib.rs:1-13`, `pyscf-kernels/src/eval_gto.rs:1-12`)
**Apply to:** every new `pyscf-dft`/`pyscf-grids` module

Each module doc-comment names: the upstream `pyscf/*.py` source file + line range being ported, the deciding D-number, the Apache-2.0 provenance for ports, and any Pitfall it owns. This is load-bearing for the bit-exact audit trail — the planner should require it in every plan's action section.

### PyO3 wall + `Python::attach` per-hook seam

**Source:** `crates/pyscf-py/src/lib.rs:13-16` (wall doc) + `crates/pyscf-py/src/bridge.rs:55-70` (`call_hook`) + `Python::attach` per hook
**Apply to:** all PyRKS/PyUKS hooks; the `KsOverrideHooks` bridge impl

`pyscf-dft` stays pyo3-free; ONLY `pyscf-py` names pyo3. Every hook dispatch is `Python::attach(|py| { ... slf.call_method1(...) })` so subclass overrides resolve via MRO (Pitfall 7). D-02's string-only `define_xc_` preserves this seam (no per-block Python callback).

---

## No Analog Found

Files requiring a fresh port of the upstream Python algorithm (no Rust analog exists; use RESEARCH Code Examples + the upstream source, with `eval_gto.rs`/`auxbasis.rs` as *style* references):

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `crates/pyscf-grids/src/lebedev.rs` | utility (generator) | transform | Lebedev `SphGenOh` generator is a fresh port of `pyscf/dft/LebedevGrid.py:113-` (D-06). No existing Rust grid code. Style analog: `eval_gto.rs` algorithm-port header + `auxbasis.rs` const tables. |
| `crates/pyscf-dft/src/parser/libxc.rs` | utility (parser) | transform | `parse_xc` is a fresh port of `pyscf/dft/libxc.py:491-718`. No existing Rust parser of this shape. Style analog: Phase 2 `format_atom`/`make_env` ports. |

Both are *style*-covered by the port + const-table conventions above; the byte-exact reference is the cited upstream Python source.

## Metadata

**Analog search scope:** `crates/pyscf-scf/`, `crates/pyscf-df/`, `crates/pyscf-diis/`, `crates/pyscf-chkfile/`, `crates/pyscf-py/`, `crates/pyscf-kernels/`, `crates/pyscf-algebra/`, `crates/pyscf-gto/`, `crates/pyscf-core/`, `python/pyscf/`, `xtask/`, workspace `Cargo.toml`; sibling-crate API surfaces (read-only) `~/Documents/workspace/{libxc_rs,xcfun_rs}`
**Files scanned:** 30
**Pattern extraction date:** 2026-05-22

# Phase 2: GTO - Research

**Researched:** 2026-05-10
**Domain:** Molecular structure (`Mole`), basis-set loading (5×11 input grid), libcint flat-array projection (`_atm`/`_bas`/`_env`), `mol.intor` over cintx, `eval_gto` cubecl kernel, ECP loader + trait shim
**Confidence:** HIGH on the canonical algorithms (every shape decision is read directly from upstream `pyscf/gto/mole.py` + cintx-compat slot constants, both of which are checked into this repo); MEDIUM on the cubecl `eval_gto` kernel sketch (no precedent kernel for grid AO evaluation in cintx-cubecl/xcfun-kernels — extrapolated from the documented vector-add / matmul launch shapes); MEDIUM on `mol.cart` ↔ `KAPPA_OF` slot-threading because upstream PySCF's libcint convention is "kappa=0 means use the C/sph code path" rather than a literal kappa value, and that contract has to be honored verbatim.

---

## Summary

Phase 2 is mostly a **port-the-canonical-algorithm** phase, not a research-the-unknown phase. The five algorithms that gate every Phase 2 success criterion (`format_atom`, `format_basis`, `make_atm_env`, `make_bas_env`, `make_env`, `make_ecp_env`) all live in `pyscf/gto/mole.py` lines 320–1160 in this repo and are Apache-2.0 — they port verbatim. The 184 builtin basis files stay where they are (D-01); a Rust `OnceLock<HashMap>` of `ParsedBasis` lazy-loads on first reference. The 11 input-form fan-out is a `match` in a single function (`Mole::format_basis`), not eleven separate code paths.

`mol.intor(name)` is a thin layer over `cintx_rs::SessionRequest` — name-string → `OperatorId` lookup, then translate `IntegralTensor.owned_values` + extents back to PySCF's array shape (F-order for most names; per-name flag, look it up in upstream `_INTOR_FUNCTIONS`). The cintx safe-facade owns the heavy lifting (workspace planning, chunk scheduling, libcint-policy gates).

`eval_gto` is the only **new** numerical kernel in this phase. It does not exist in cintx (cintx is integral evaluation; `eval_gto` is AO-on-grid evaluation, used by DFT in Phase 4). The kernel lives in `pyscf-kernels` per D-04, launched through `pyscf-algebra` per the algebra wall, with `pyscf-gto::eval_gto` as the user-facing wrapper. Phase 2 ships the `GTOval` and `GTOval_sph` (single-component AO eval) variants as MVP; the four derivative variants (`_deriv1`, `_deriv2`, `_ip`, `_ig`) ship as stubs returning `NotYetImplemented { phase: 4, what: "..." }` if Phase 2 size is tight (Claude's discretion item).

ECP per D-06 ships **loading only**: the NWChem-format ECP basis-file parser (port of `pyscf/gto/basis/parse_nwchem_ecp.py`) + `EcpEngine` trait + `EcpEngineNotAvailable` stub. Evaluation (`int1e_ecp` family) closes via a follow-up gap-closure plan (analogous to Phase 1 `01-08-PLAN.md`) when the cintx ECP workstream lands cint1e_ecp Type-1 + Type-2 projectors.

**Primary recommendation:** Wave 0 = `cintx-rs` ↔ `pyscf-gto` integration smoke test (proves the path-dep cintx is reachable from pyscf-gto and `SessionRequest` round-trips a trivial overlap integral) + `pyscf-kernels` `cubecl-cpu` minimal launch (proves the kernel/algebra wiring works) BEFORE the basis-loader and `make_env` ports start. These are the two non-port risks; everything else is mechanical translation of well-tested Python.

---

## User Constraints (from CONTEXT.md)

### Locked Decisions

> Verbatim from `02-CONTEXT.md` `<decisions>` block.

**Basis file packaging (Area 1)**
- **D-01:** Read built-in basis files live from upstream `pyscf/gto/basis/` at runtime. No `build.rs` codegen, no `include_bytes!` snapshot — parsing is lazy at first use, behind a `OnceLock<HashMap<String, ParsedBasis>>` cache. Constraint: chosen because of the project-wide "don't freeze compile" preference (heavy build-time parsing rejected).
- **D-02:** Path resolver uses a priority chain: (1) `PYSCF_BASIS_PATH` env var if set, (2) repo-relative walk-up from `CARGO_MANIFEST_DIR` / `current_exe()` looking for `../../pyscf/gto/basis/`, (3) error with a message naming the env var. Mirrors the `PYSCF_BACKEND` env-var pattern locked in Phase 1 D-07.

**Mole ↔ cintx bridge (Area 2)**
- **D-03:** `Mole.build()` eagerly projects `cintx_core::BasisSet` (typed, `Arc`-shared, single source of truth — satisfies GTO-11 zero-copy) into derived flat arrays cached on `Mole`: `_atm: Vec<i32>`, `_bas: Vec<i32>`, `_env: Vec<f64>`, `ao_loc_nr: Vec<i32>`, `nao_nr: usize`. Eager (not lazy) so byte-identity assertions against upstream are direct `assert_eq!` checks, and `build()` semantics match upstream PySCF's "materialize everything" contract. Reuse `cintx-compat::raw` slot constants (`ATM_SLOTS=6`, `BAS_SLOTS=8`, `PTR_ENV_START=20`) — do **not** duplicate libcint slot layout knowledge.

**eval_gto kernel home (Area 3)**
- **D-04:** The `eval_gto` cubecl kernel lives in `pyscf-kernels`; `pyscf-gto::eval_gto(mol, name, coords)` is the user-facing wrapper that builds the shell list and dispatches via `pyscf-algebra` (algebra-wall friendly — `pyscf-gto` never imports `cubecl-*`). Mirrors the cintx-cubecl / xcfun-kernels split established in Phase 1 D-04. Phase 4 DFT imports `pyscf-kernels` directly for grid loops (not via `pyscf-gto`).

**ECP scope (Area 4)**
- **D-05:** `int1e_ecp` (and ECP gradient variants for Phase 7 GRAD-07) belong in cintx, not in pyscf-rs. cintx is the libcint replacement; the integral engine seam is the right place for ECP.
- **D-06:** Sequencing is **parallel**, not blocking. Phase 2 ships:
  1. ECP basis-file parser in `pyscf-gto` (parses NWChem-format ECP blocks from `.dat` files like LANL2DZ, SBKJC).
  2. `EcpEngine` trait declared in `pyscf-core` (D-07).
  3. Stub impl that returns `EcpEngineNotAvailable` until cintx ECP lands.

  A separate cintx workstream (out of pyscf-rs scope) ships `cint1e_ecp` Type-1 + Type-2 projectors. When that lands, a Phase 2 gap-closure plan (analogous to Phase 1's `01-08-PLAN.md`) bumps the cintx pin + wires the cintx-side `EcpEngine` impl. GTO-05 effectively splits into "loading shipped in Phase 2" + "evaluation closed by gap-closure plan."
- **D-07:** `EcpEngine` is a separate trait in `pyscf-core` (not an extension to `IntegralEngine`). Surface includes `ecp_int1e(&self, mol, basis) -> Result<...>` plus `_ipnuc` variants for Phase 7 ECP gradients. The user-facing API stays uniform — `mol.intor("int1e_ecp")` — because pyscf-gto's `intor` dispatcher routes `int1e_ecp*` names to `EcpEngine` internally. Cintx implements both `IntegralEngine` and `EcpEngine` on the same engine type; non-ECP cintx versions simply don't impl `EcpEngine`.

### Claude's Discretion

The researcher / planner picks the implementation for these (not user-decided):

- **ALIAS table porting** — hand-port upstream's `pyscf/gto/basis/__init__.py::ALIAS` dict (395 entries verified, see "Standard Stack" below) to a Rust `static OnceLock<HashMap<&'static str, &'static str>>`. Source-of-truth lives in the Python module; oracle test catches drift.
- **`USER_BASIS_DIR` / `USER_BASIS_ALIAS` override semantics** — upstream supports user-supplied basis dirs and alias overrides via `pyscf.__config__`. Planner picks whether to surface these as `PYSCF_USER_BASIS_DIR` env vars, a programmatic API on `Mole`, or both.
- **Parser-dispatch shape** — upstream routes through `parse_nwchem`, `parse_nwchem_ecp`, `parse_cp2k`, `parse_cp2k_pp`. Planner picks whether to mirror as 4 separate Rust modules, one dispatcher with format-detect, or a single parser with format-aware tokens.
- **`mol.cart` flag handling** — spherical (default) vs cartesian AO counting in the projection: `(2l+1)` vs `(l+1)(l+2)/2` per shell. Both modes must work. Planner picks how to thread the flag through `_bas` `kappa` slot.
- **`Arc<BasisSet>` vs by-value ownership of cintx structure inside Mole** — `Arc` is the obvious answer given GTO-11 zero-copy intent, but planner confirms.
- **`mol.set_geom_(new_atom)` cache invalidation** — when geometry mutates, `_atm` rows must update but `_bas`/basis structure is preserved. Planner picks a granular invalidation strategy.
- **6 GTOval variants priority for Phase 4 DFT** — `GTOval_sph` and `GTOval_sph_deriv1` are the hot paths for DFT grid integration; `GTOval_ig`/`GTOval_ip` are gradient-side. Planner can ship in priority order if Phase 2 size is tight.
- **Output layout (F-order vs C-order)** — match upstream PySCF's per-integral convention (most are F-order). Planner consults `pyscf/gto/moleintor.py` per-name for the truth and documents in code.
- **Grid-batching strategy for cubecl dispatch** — eval_gto over 10000 grid points × 100 AOs is one big kernel launch vs chunked. Planner consults `docs/manual/Cubecl/` and the cubecl-matmul precedent.
- **Atom-input "callable" form (5th form)** — deferred to Phase 3 (needs PyO3); Phase 2 stub returns `NotYetImplemented { phase: 3, what: "atom callable form (GTO-01.5)" }`.
- **`mol.dumps()` / `gto.Mole.loads()` format** — semantic round-trip + oracle interop tested (PySCF writes → pyscf-rs reads + reverse). Not byte-identical to upstream's JSON string formatting (fragile, GTO-09 doesn't require it). Planner picks `serde_json` vs hand-rolled writer.
- **Test corpus tiering** — small (H2O/cc-pVDZ + benzene/6-31G* + water-trimer) for PR-CI, full 5×11 atom×basis grid for nightly. Phase 8 already has a per-basis sweep slot (ORACLE-06).

### Deferred Ideas (OUT OF SCOPE)

- **`int1e_ecp` evaluation kernel inside cintx** — separate cintx workstream (Type-1 + Type-2 projectors); pyscf-rs gap-closure plan wires when available (D-06).
- **Atom-input "callable" (5th form)** — needs PyO3 (Phase 3 BIND-02). Phase 2 returns `NotYetImplemented { phase: 3, what: "atom callable form" }`.
- **Wheel packaging of `pyscf/gto/basis/`** — D-01 reads at runtime, so the maturin wheel must bundle these files (~MB). Phase 8 DIST-02 owns the wheel-content manifest.
- **`mol.dumps()` byte-identical to upstream's JSON string** — explicitly out of scope; semantic round-trip + oracle interop is the contract (Claude's discretion).
- **Phase 4 DFT GTOval variant priority** — `GTOval_sph` + `GTOval_sph_deriv1` are the hot paths; ranking can defer to Phase 4 plan if Phase 2 size is tight.
- **`USER_BASIS_DIR`/`USER_BASIS_ALIAS` config-dict overrides** — Phase 2 may stub; Phase 3 PyO3 integration is where users actually set these.
- **Per-basis nightly sweep** — already owned by Phase 8 ORACLE-06; Phase 2 only commits to the small PR-CI corpus.
- **libxc_rs re-enable** — Phase 4 (DFT-03 routes through libxc_rs).

---

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| GTO-01 | `pyscf.M(...)` factory + `gto.Mole` accept all 5 atom-input forms | `format_atom` algorithm at `pyscf/gto/mole.py:320-415` ports verbatim. 4 forms ship in Phase 2; "callable" (5th) returns `NotYetImplemented { phase: 3 }`. See "Code Examples" §1. `[VERIFIED: file]` |
| GTO-02 | `mol.basis = ...` accepts all 11 input forms | `format_basis` at `mole.py:418-466` + `_generate_basis_converter` at `mole.py:468-506`. The 11 forms collapse to 4 categories: by-name (string), per-element dict, parsed text (NWChem/CP2K), and explicit nested-list. Full enumeration in "Architecture Patterns" §2. `[VERIFIED: file]` |
| GTO-03 | All 207 built-in basis-set files resolve correctly + `gto.parse(...)` accepts user text | **Verified count: 184 unique `*.dat` files** in `pyscf/gto/basis/` excluding Zone.Identifier copies; **305 if counting subdirs** (`pople-basis/`, `ccecp-basis/`, `dyall-basis/`, `f12-basis/`, `soecp/`). The "207" figure in CONTEXT.md is approximate but in-range. `ALIAS` dict has **395 entries** (also verified by grep). D-01 + D-02 path resolver covers them. `[VERIFIED: filesystem]` |
| GTO-04 | `mol.build()` produces identical `_atm`/`_bas`/`_env`/`ao_loc_nr`/`nao_nr` to upstream byte-for-byte | `make_env` algorithm at `mole.py:1029-1105`; reuses `cintx-compat::raw` slot constants per D-03. Direct port — no algorithmic invention. `make_loc` algorithm at `pyscf/gto/moleintor.py:804-820` for `ao_loc_nr`. See "Code Examples" §3. `[VERIFIED: file]` |
| GTO-05 | ECP loading + `int1e_ecp` evaluation match upstream bit-exact | **SPLIT per D-06**: loading + `EcpEngine` trait + stub ship in Phase 2 (parser ports `pyscf/gto/basis/parse_nwchem_ecp.py`; algorithm at `mole.py:1107-1160` for `make_ecp_env`). Evaluation closes via gap-closure plan when cintx ships `cint1e_ecp` Type-1 + Type-2. `[VERIFIED: file]` |
| GTO-06 | `mol.intor(name, ...)` is a thin wrapper over cintx | Build `SessionRequest { operator: lookup(name), representation, basis: &cintx_basis, shells, options }` from `cintx-rs/src/api.rs:17-79`. The mapping `intor_name → OperatorId` is the only Phase 2 invention; everything else delegates. Layout-translation is the F-order subtlety (see Pitfall 1 below). `[VERIFIED: file]` |
| GTO-07 | `eval_gto(mol, eval_name, coords, ...)` for 6 variants | Reference: `pyscf/gto/eval_gto.py` (248 lines, full algorithm). The cubecl kernel design is in "Architecture Patterns" §4. Per Claude's discretion: `GTOval_sph` is the priority MVP; deriv variants stub if needed. `[VERIFIED: file]` |
| GTO-08 | `Mole` exposes the ≥30 attribute floor | Enumerated in "Standard Stack" §"Mole Attribute Floor" below. `[VERIFIED: file]` (sourced from `MoleBase` class at `mole.py:2189+`) |
| GTO-09 | `mol.dumps()` / `mol.loads()` JSON round-trip; `mol.copy()` deep-copy | `dumps`/`loads` at `mole.py:1251-1349`. Per Claude's discretion: semantic round-trip only, not byte-identical to upstream's JSON string. Use `serde_json` with `mol.atom`/`basis`/`ecp` serialized as `Display`/`repr`-style strings (matches upstream's `repr(mol.atom)`). `[VERIFIED: file]` |
| GTO-10 | `mol.set_geom_(new_atom)` mutates in place, returns `self` | Granular cache invalidation: re-run `make_atm_env` for new coordinates, **preserve** `_bas` and the parsed basis since geometry doesn't change basis structure. `_env` updates in-place at the coordinate slots only (PTR_COORD entries). See "Architecture Patterns" §5. `[VERIFIED: file]` (`set_geom_` exists upstream at `mole.py` later in the file) |
| GTO-11 | `pyscf-core::BasisSet` re-exports `cintx_core::BasisSet` zero-copy | Replace Phase 1 placeholder `crates/pyscf-core/src/basis_set.rs` with `pub use cintx_core::BasisSet;` — `Arc<[Atom]>`/`Arc<[Arc<Shell>]>` already inside cintx_core::BasisSet means re-export is literally zero-copy. `[VERIFIED: file]` (cintx structure at `cintx/crates/cintx-core/src/basis.rs:48-110`) |

---

## Project Constraints (from CLAUDE.md)

**No `./CLAUDE.md` file exists at the repo root.** No project-specific directives to enforce beyond what is already encoded in:

- `.planning/STATE.md` Blockers/Concerns (cubecl 0.10.0 lockstep, cintx ECP coordination)
- `.planning/PROJECT.md` Constraints (cubecl is sole compute primitive; algebra wall; bit-exact contract; Apache-2.0)
- `~/.claude/projects/-home-user-Documents-workspace-pyscf-rs/memory/` (libxc_rs ~6h compile — disabled in `[patch.crates-io]`; "don't freeze compile" — no heavy build.rs)
- Phase 1 D-01..D-15 (workspace layout, AlgebraClient, env-var pattern, sibling-crate sourcing)

The Phase 1 D-01..D-15 constraints carry forward to Phase 2 and are treated as locked.

---

## Standard Stack

### Core (verified 2026-05-10)

| Crate / Resource | Version / Source | Purpose | Why Standard |
|---|---|---|---|
| `cintx-core::BasisSet` | path: `../cintx`, current `[patch.crates-io]` (Cargo.toml:93) | Typed `Arc`-backed `BasisSet { atoms: Arc<[Atom]>, shells: Arc<[Arc<Shell>]>, meta: BasisMeta }` (`cintx/crates/cintx-core/src/basis.rs:48-52`). pyscf-core re-exports zero-copy per GTO-11. | Single source of truth for shell/atom structure across cintx, libxc_rs, xcfun_rs, pyscf_rs. Hand-rolling a parallel structure violates the four-crate consistency contract. `[VERIFIED: file]` |
| `cintx-compat::raw` slot constants | path: `../cintx`, same patch | `ATM_SLOTS=6`, `BAS_SLOTS=8`, `PTR_ENV_START=20`, `CHARGE_OF=0`, `PTR_COORD=1`, `NUC_MOD_OF=2`, `ATOM_OF=0`, `ANG_OF=1`, `NPRIM_OF=2`, `NCTR_OF=3`, `KAPPA_OF=4`, `PTR_EXP=5`, `PTR_COEFF=6`. (`cintx/crates/cintx-compat/src/raw.rs:15-41`) | Direct mirror of libcint's `cint_bas.h`; reusing avoids duplicating layout knowledge per D-03. Drift between cintx-compat and pyscf-rs would silently break byte-identity. `[VERIFIED: file]` |
| `cintx-rs::SessionRequest` / `SessionQuery` / `IntegralTensor` | path: `../cintx` | User-facing safe API (`cintx/crates/cintx-rs/src/api.rs:17-454`). `mol.intor(name)` constructs `SessionRequest::new(operator, representation, &basis, shells, ExecutionOptions::default())` then `request.query_workspace()?.evaluate()`. Returns `IntegralTensor { extents, owned_values, ... }`. | Already-built workspace planner, chunk scheduler, libcint policy gate. Phase 2 must NOT reinvent these. `[VERIFIED: file]` |
| `pyscf-algebra` (Phase 1) | workspace member, current path | `AlgebraClient` enum (`crates/pyscf-algebra/src/client.rs:10-18`), `Tensor`/`BufferId` opaque surface (`tensor.rs:1-40`). `pyscf-kernels::eval_gto` launches via this — never imports `cubecl-*` directly (algebra wall ALG-06). | Algebra wall enforced by xtask lint; deviation is a CI-blocking failure. `[VERIFIED: file]` |
| `cubecl` (eval_gto kernel only) | `=0.10.0` (workspace pin, Cargo.toml:34) | `#[cube(launch_unchecked)]` for the eval_gto kernel inside `pyscf-kernels`. | The pinned-lockstep workspace policy from Phase 1 D-12. **Note**: cubecl-matmul / cubecl-reduce are pinned at `=0.9.0-pre.5` (workspace Cargo.toml:44-45) due to known 0.10.0 publish gap — Phase 2 eval_gto does not need either, only base `cubecl` runtime. `[VERIFIED: file]` |
| `serde` / `serde_json` | `=1.0` / `=1.0.149` (workspace) | `mol.dumps()`/`loads()` JSON round-trip per GTO-09 (Claude's discretion: semantic, not byte-identical). | Already in workspace; no new dependency. `[VERIFIED: file]` |
| `thiserror` | `=2.0.18` (workspace) | `BasisLoadError`, `EcpLoadError`, `EcpEngineNotAvailable`. | Already in workspace; matches Phase 1 error pattern (`pyscf-core/src/error.rs`). `[VERIFIED: file]` |
| Upstream `pyscf/gto/basis/` files | already in repo, runtime path resolution | 184 unique `*.dat` files at top level, plus subdirs `pople-basis/` (Pople ext'd polarisation files), `ccecp-basis/`, `dyall-basis/`, `f12-basis/`, `soecp/`. ALIAS dict has **395 entries** mapping basis-name → file. | D-01 + D-02 lock this. **Do not vendor or copy** — runtime resolution per `PYSCF_BASIS_PATH`. `[VERIFIED: filesystem]` |

### Supporting

| Crate | Version | Purpose | When to Use |
|---|---|---|---|
| `bytemuck` | `=1` features=`["derive"]` (workspace) | Cast `Vec<f64>`/`&[i32]` ↔ raw bytes for cubecl `client.create(...)` uploads in eval_gto. | Already in pyscf-algebra; pyscf-kernels picks up via the algebra wall. |
| `tracing` | `=0.1.44` (workspace) | Verbosity logging at basis-loader (`tracing::debug!("loading basis {} from {}", name, path)`) and intor dispatch (`tracing::trace!("intor route: {} → {}", name, route)`). | Mirrors Phase 1 D-09 (FOUND-09 Python-verbosity contract); Phase 2 hooks the call sites, Phase 3 wires `mol.verbose` → `tracing-subscriber` filter. |
| `approx` | `=0.5.1` dev-dep (workspace) | Float-tolerance assertions in unit tests where bit-identity is loosened (e.g., normalisation factors after the `_nomalize_contracted_ao` matrix inversion). | Test code only. |
| `rstest` | `=0.26.1` dev-dep (workspace) | Parametric tests over the 5×11 atom×basis grid. | Test code only. |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|---|---|---|
| `OnceLock<HashMap<String, ParsedBasis>>` (D-01) | `dashmap` | DashMap brings concurrent insertion but Phase 2 has no concurrent insertion path; `OnceLock` is std and cheaper. Reject. |
| Re-export `cintx_core::BasisSet` | Newtype `pyscf_core::BasisSet(cintx_core::BasisSet)` | Newtype loses the zero-copy contract (GTO-11 explicit). Reject. |
| `serde_json` for `mol.dumps()` | Hand-rolled writer (literal-match upstream JSON) | Upstream uses Python `repr()` for nested fields then `json.dumps()` — not byte-stable across Python versions. Hand-rolling that is a fragile maintenance burden, GTO-09 doesn't require it (Claude's discretion notes this explicitly). Use `serde_json` + Display-impl on `mol.atom`/`basis`/`ecp` for upstream-compatibility on the **content**, not the wire format. |
| Single `parse_basis_text` with format detection | 4 separate `parse_nwchem`/`parse_nwchem_ecp`/`parse_cp2k`/`parse_cp2k_pp` modules (mirror upstream) | Mirroring upstream gets cheaper porting (function-by-function), simpler code review, and matches the canonical references list. Single-function with format-detect is more idiomatic Rust but loses the line-by-line port advantage. **Recommendation: mirror upstream** (4 modules). Discretion-locked — planner picks. |
| `PYSCF_USER_BASIS_DIR` env var only | Add a `Mole::set_user_basis_dir(p)` method too | Upstream uses `pyscf.__config__.USER_BASIS_DIR` which is module-import-time; Rust equivalent is the env var. The programmatic API is a Phase 3 PyO3 concern. **Recommendation: env var in Phase 2, Phase 3 adds programmatic surface.** Discretion-locked. |

### Mole Attribute Floor (≥30 — for GTO-08)

Required by GTO-08. Sourced from `pyscf/gto/mole.py` `MoleBase` class (line 2189+) and `format_atom`/`format_basis`/`make_env` outputs. Phase 2 implements **all** of these on `pyscf-core::Mole`:

| # | Attribute | Type | Source / Notes |
|---|---|---|---|
| 1 | `atom` | `String` (raw) or `Vec<AtomInputForm>` | User input — preserved verbatim |
| 2 | `basis` | `String` or `BasisInputForm` | User input — preserved verbatim |
| 3 | `charge` | `i32` | `mole.py:2189+` |
| 4 | `spin` | `i32` (multiplicity − 1) | Existing Phase 1 stub |
| 5 | `nelectron` | `usize` | `tot_electrons` at `mole.py:1162-1186` |
| 6 | `natm` | `usize` | `_atm.len() / ATM_SLOTS` |
| 7 | `nbas` | `usize` | `_bas.len() / BAS_SLOTS` |
| 8 | `nao_nr` | `usize` | `nao_nr` at `mole.py:1378-1385` (sph default) |
| 9 | `nao_2c` | `usize` | `nao_2c` at `mole.py:1418` (spinor) — Phase 2 stub OK; rare |
| 10 | `ao_loc_nr` | `Vec<i32>` | `ao_loc_nr` at `mole.py:1454-1471` (cumsum of dims) |
| 11 | `ao_labels` | `Vec<String>` | `ao_labels` at `mole.py:1656` — defer body, expose method that constructs on demand |
| 12 | `cart` | `bool` | `false` default = spherical AOs; `true` = Cartesian (mol.cart) |
| 13 | `verbose` | `u8` (0–9) | Phase 3 wires to `tracing-subscriber`; Phase 2 stores the value |
| 14 | `max_memory` | `f64` (MB) | Used by CCSD-08 in Phase 6; Phase 2 stores |
| 15 | `unit` | `enum { Ang, Bohr, AU }` | `format_atom` parses `unit=` kwarg |
| 16 | `output` | `Option<PathBuf>` | Logging output file (Phase 3 wires to `tracing` writer); Phase 2 stores |
| 17 | `_atm` | `Vec<i32>` length `natm * 6` | `make_env` (D-03) |
| 18 | `_bas` | `Vec<i32>` length `nbas * 8` | `make_env` (D-03) |
| 19 | `_env` | `Vec<f64>` | `make_env` (D-03) |
| 20 | `_ecpbas` | `Vec<i32>` length `n_ecp_shells * 8` | `make_ecp_env` (D-06 loading half) |
| 21 | `_atom` | `Vec<(String, [f64;3])>` (internal symbol+Bohr coord pairs) | `format_atom` output |
| 22 | `_basis` | `HashMap<String, ParsedBasis>` | `format_basis` output |
| 23 | `_ecp` | `HashMap<String, ParsedEcp>` | `make_ecp_env` precursor |
| 24 | `_built` | `bool` | `mol.build()` was called |
| 25 | `nucmod` | `HashMap<String, NuclearModel>` | `make_atm_env` per-atom override |
| 26 | `nucprop` | `HashMap<String, f64>` | Optional finite-nucleus zeta |
| 27 | `symmetry` | `bool` | False in v1 (out of scope per CONTEXT-1 forbidden-paths); store as `false` always |
| 28 | `groupname` | `String` | "C1" always in v1 |
| 29 | `topgroup` | `String` | "C1" always in v1 |
| 30 | `enuc` | `f64` (lazy) | `classical_coulomb_energy` at `mole.py:1522` (computed on demand) |
| 31 | `ecp` | `String` or `EcpInputForm` | User input — preserved verbatim |
| 32 | `pseudo` | `Option<...>` | Out of scope (PBC); `None` always |
| 33 | `mass` | `Vec<f64>` (lazy) | `atom_mass_list` at `mole.py:1992` |

The 30-attribute floor is **easily exceeded** — listing 33 above with the upstream lineage. Planner is free to expand the surface. The methods (`atom_charges()`, `atom_coords()`, `atom_coord(i)`, `bas_angular(i)`, `bas_nctr(i)`, `bas_nprim(i)`, `intor(name)`, `eval_gto(name, coords)`, `dumps()`, `loads(s)`, `copy()`, `set_geom_(new_atom)`, `analyze()` (stub), …) are method-floor obligations and are documented in the plan, not the attribute floor.

---

## Architecture Patterns

### Recommended Project Structure (Phase 2 additions only)

```
crates/
├── pyscf-core/
│   └── src/
│       ├── basis_set.rs     # REPLACE placeholder → `pub use cintx_core::BasisSet;` (GTO-11)
│       ├── traits.rs         # ADD `EcpEngine` trait per D-07 (alongside existing IntegralEngine)
│       ├── error.rs          # ADD variants: BasisLoadError, EcpLoadError, EcpEngineNotAvailable
│       └── mole.rs           # FILL ≥30-attribute floor (GTO-08), add ParsedBasis/ParsedAtom types
├── pyscf-gto/
│   └── src/
│       ├── lib.rs            # public surface: pyscf.M(...) factory, gto::Mole re-export
│       ├── format_atom.rs    # GTO-01: 4 atom-input forms (5th = NotYetImplemented{phase:3})
│       ├── format_basis.rs   # GTO-02: 11 basis-input forms collapsed to a match
│       ├── basis/            # parser modules (mirror upstream layout)
│       │   ├── mod.rs        # ALIAS table OnceLock<HashMap<&str, &str>> (~395 entries)
│       │   ├── path.rs       # D-02 PYSCF_BASIS_PATH resolver
│       │   ├── nwchem.rs     # port of pyscf/gto/basis/parse_nwchem.py
│       │   ├── nwchem_ecp.rs # port of pyscf/gto/basis/parse_nwchem_ecp.py (D-06 loading)
│       │   ├── cp2k.rs       # port of pyscf/gto/basis/parse_cp2k.py
│       │   └── cp2k_pp.rs    # port of pyscf/gto/basis/parse_cp2k_pp.py
│       ├── make_env.rs       # D-03: make_atm_env + make_bas_env + make_env + make_ecp_env
│       ├── intor.rs          # GTO-06: mol.intor(name) → cintx_rs::SessionRequest
│       ├── eval_gto.rs       # GTO-07: user wrapper, dispatches via pyscf-algebra
│       ├── ecp_engine_stub.rs # D-06 stub returning EcpEngineNotAvailable
│       └── dumps_loads.rs    # GTO-09: serde_json round-trip
├── pyscf-kernels/
│   └── src/
│       ├── lib.rs
│       └── eval_gto.rs       # D-04: cubecl #[cube(launch_unchecked)] AO-on-grid kernel
```

**Crates that get NEW dependencies** (planner adds to Cargo.toml):
- `pyscf-gto/Cargo.toml`: `pyscf-core`, `pyscf-algebra`, `pyscf-kernels`, `cintx-core`, `cintx-compat`, `cintx-rs`, `serde`, `serde_json`, `thiserror`, `tracing`. **No `cubecl-*`** (algebra wall).
- `pyscf-kernels/Cargo.toml`: `pyscf-core`, `pyscf-algebra` (re-exports `Tensor`), `cubecl`, `cubecl-cpu` (default), `cubecl-cuda`/`cubecl-wgpu`/`cubecl-hip` (optional, feature-gated), `bytemuck`, `tracing`. **No `cintx-*`** (cintx is for integrals, eval_gto is independent).
- `pyscf-core/Cargo.toml`: ADD `cintx-core` (for re-export per GTO-11). No other new deps.

### Pattern 1: Lazy basis loader with priority-chain path resolution (D-01 + D-02)

**What:** First call to `load_basis(name, symbol)` resolves the basis-data directory once (`OnceLock`), then resolves the basis filename via ALIAS lookup, parses NWChem/CP2K text, and caches the parsed shells in a `OnceLock<HashMap<(name, symbol), ParsedBasis>>`. Subsequent calls hit cache.

**When to use:** Always (D-01 mandates it).

**Example:**
```rust
// pyscf-gto/src/basis/path.rs
use std::path::PathBuf;
use std::sync::OnceLock;

static BASIS_DIR: OnceLock<PathBuf> = OnceLock::new();

pub fn basis_dir() -> Result<&'static PathBuf, BasisLoadError> {
    BASIS_DIR.get_or_init(|| {
        // D-02 priority chain:
        // (1) PYSCF_BASIS_PATH env override
        if let Ok(p) = std::env::var("PYSCF_BASIS_PATH") {
            let path = PathBuf::from(p);
            if path.is_dir() { return path; }
        }
        // (2) walk-up from CARGO_MANIFEST_DIR / current_exe() looking for ../../pyscf/gto/basis/
        let candidates = [
            // workspace-relative, dev mode
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../pyscf/gto/basis"),
            // installed wheel relative
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../pyscf/gto/basis"),
        ];
        for c in &candidates {
            if c.is_dir() { return c.canonicalize().unwrap_or_else(|_| c.clone()); }
        }
        // (3) fall through — caller error path will format the message
        PathBuf::new()
    });
    let p = BASIS_DIR.get().unwrap();
    if p.as_os_str().is_empty() {
        Err(BasisLoadError::PathNotFound {
            tried: "PYSCF_BASIS_PATH env, CARGO_MANIFEST_DIR walk-up".into(),
        })
    } else {
        Ok(p)
    }
}
```

### Pattern 2: 11-form basis input collapsed to a single match (GTO-02)

**What:** Upstream `format_basis` accepts 11 input forms. They collapse to 4 dispatch arms in Rust:

```rust
// pyscf-gto/src/format_basis.rs
pub enum BasisInput {
    Name(String),                           // "cc-pvdz"
    PerElement(HashMap<String, BasisInput>), // {"H": "sto-3g", "O": "cc-pvdz"}
    NwchemText(String),                      // raw NWChem-format string
    Cp2kText(String),                        // raw CP2K-format string (detected by "GTH")
    Parsed(ParsedBasis),                     // already-parsed nested list
}

impl BasisInput {
    pub fn from_str_smart(s: &str) -> BasisInput {
        // Mirror upstream's load() at pyscf/gto/basis/__init__.py:621-714 in priority order:
        // 1. file path → load_external (sniff format)
        // 2. ALIAS lookup → load NWChem
        // 3. GTH_ALIAS → load CP2K
        // 4. _is_pople_basis → _parse_pople_basis
        // 5. raw text with "GTH" → CP2K
        // 6. raw text with "ECP" → NWChem ECP
        // 7. raw text otherwise → NWChem (with fallback to CP2K)
        // ...
    }
}
```

The 11 upstream forms are:
1. Built-in name (string): `"cc-pvdz"` → ALIAS lookup
2. Pople ext name: `"6-31g**"`, `"6-311+g*"` → `_parse_pople_basis`
3. Per-element dict: `{"H": "sto-3g", "O": "cc-pvdz"}`
4. ECP-bcc style: `"bfd-vdz"` (ECP-fitted basis names — same as #1 mechanism, different ALIAS)
5. F12 basis: `"cc-pvdz-f12"` → resolves to `f12-basis/` subdir
6. Dyall (rel'tic): `"dyalldz"` → resolves to `dyall-basis/` subdir
7. ANO contractions: `"ano"`, `"anoroosdz"` → ALIAS
8. def2 family: `"def2-svp"`, `"def2-tzvp"` → ALIAS
9. Parsed Gaussian-94 text: raw-string passed through `parse_nwchem.parse(...)` (Gaussian-94 is upstream's "NWChem format" effectively)
10. NWChem text directly: same path
11. Auto-segmented (general → segmented): `optimize_contraction(b)` post-process flag

The Rust enum collapses 1+2+4+5+6+7+8 into `BasisInput::Name(..)`-with-ALIAS-routing; 3 into `PerElement`; 9+10 into `NwchemText`; 11 is a post-processing flag (`optimize: bool`) on the parser.

### Pattern 3: `make_env` (D-03) — flat-array projection

**What:** `Mole::build()` runs after `format_atom` + `format_basis`. It walks atoms and basis specs, produces the three flat arrays libcint expects, plus `ao_loc_nr` and `nao_nr`. Direct port of `pyscf/gto/mole.py:1029-1105`.

**Slot layout (REUSE `cintx_compat::raw` constants; do NOT redefine):**
```rust
// _atm rows are 6 i32 each (ATM_SLOTS=6):
//   [CHARGE_OF=0]    : nuc_charge (atomic number, possibly minus ECP electrons)
//   [PTR_COORD=1]    : offset into _env where xyz starts (3 doubles)
//   [NUC_MOD_OF=2]   : 0=Point, 1=Gaussian-finite, 2=ECP
//   [PTR_ZETA=3]     : offset into _env for finite-nucleus zeta (1 double)
//   slots 4,5        : reserved (zero in upstream)

// _bas rows are 8 i32 each (BAS_SLOTS=8):
//   [ATOM_OF=0]      : index into _atm rows
//   [ANG_OF=1]       : angular momentum l
//   [NPRIM_OF=2]     : number of primitives
//   [NCTR_OF=3]      : number of contractions
//   [KAPPA_OF=4]     : 0 for sph/cart, signed int for spinor (kept 0 for v1 — no spinor in scope)
//   [PTR_EXP=5]      : offset into _env where exponents start (NPRIM doubles)
//   [PTR_COEFF=6]    : offset into _env where coeff matrix starts (NPRIM*NCTR doubles, F-order)
//   slot 7           : reserved (zero)

// _env layout (PTR_ENV_START=20 first slots are libcint reserved):
//   env[0..20]       : libcint global params (PTR_EXPCUTOFF, PTR_COMMON_ORIG, PTR_RINV_ORIG,
//                      PTR_RANGE_OMEGA, AS_ECPBAS_OFFSET=18, AS_NECPBAS=19, ...)
//   env[20..]        : per-atom coords (3 each), per-basis exponents + coefficient matrices
```

**Order of writes** (reproduce upstream's exact order; otherwise `_env` indices drift):
1. Initialise `_env` with `PTR_ENV_START=20` zeros (preserves global-param slots).
2. For each atom: `make_atm_env(atom, ptr_env, ...)` → push xyz (+ zeta if Gaussian nucleus) to `_env`, `_atm` row. Update `ptr_env`.
3. **Group basis by symbol**: For each unique element symbol, run `make_bas_env(basis_for_symbol, atom_id=0, ptr_env)` once → produces a `_bas` template + `_env` block (exponents + coeff matrices). Cache template in `_basdic` keyed by symbol.
4. For each atom: clone the per-symbol template, **patch `ATOM_OF` to the actual atom index**, append to `_bas`.

The grouping in step 3 is what makes byte-identity work: upstream processes basis-data once per unique symbol, not once per atom.

**Coefficient normalisation** (`gto_norm` + `_nomalize_contracted_ao`): the contraction coefficients in `_env` are NOT the raw values from the basis file. They have two normalisation factors absorbed:
- **Per-primitive radial norm** (`gto_norm(l, expnt)` at `mole.py:125-155`): `1 / sqrt(gaussian_int(2l+2, 2*expnt))`
- **Contracted-AO normalisation** (`_nomalize_contracted_ao` at `mole.py:1018-1027`): post-multiply each contracted column by `1/sqrt(c.T @ S @ c)` where `S[i,j] = gaussian_int(2l+2, e_i+e_j)`. Controlled by `NORMALIZE_GTO=True` upstream default.

Both transforms must be applied byte-for-byte or the integral byte-identity test (Phase 2 success criterion 1) fails.

### Pattern 4: `eval_gto` cubecl kernel (D-04) — element-wise per-grid-point

**What:** A single `#[cube(launch_unchecked)]` kernel evaluates AO values at grid points. One kernel covers `GTOval` and `GTOval_sph` (Phase 2 MVP); deriv variants extend the same shape.

**Why one kernel covers both `_sph` and `_cart`:** the difference is which spherical-harmonic transform applies AFTER the radial+exponential evaluation. Cartesian = raw `x^a y^b z^c r^l e^(-α r²)`; spherical = same, then transform via `cart2sph_l` matrix. `cart2sph_l` is small (a 3×6 matrix for d-shells, e.g.) and can be pre-uploaded as a comptime constant or a small lookup buffer.

**Recommended kernel sketch** (extrapolated from `Cubecl_multi_ compute.md` vector_add pattern + `eval_gto.py`):

```rust
// crates/pyscf-kernels/src/eval_gto.rs
use cubecl::prelude::*;

/// Per-grid-point AO evaluation.
/// One thread per (grid_point, ao_index) pair.
/// Output shape: ngrids × nao (F-order via index calc).
#[cube(launch_unchecked)]
fn eval_gto_sph_kernel(
    coords: &Array<f64>,        // ngrids * 3, F-order (so coords[i+0*ngrids]=x[i] etc.)
    atm: &Array<i32>,           // natm * 6, libcint _atm array
    bas: &Array<i32>,           // nbas * 8, libcint _bas array
    env: &Array<f64>,           // PTR_ENV_START + per-atom coords + per-basis exp/coeff
    ao_loc: &Array<i32>,        // nbas+1
    out: &mut Array<f64>,       // ngrids * nao, F-order
    #[comptime] ngrids: u32,
    #[comptime] nao: u32,
) {
    let g = ABSOLUTE_POS;       // grid index
    if g >= ngrids { return; }
    // For each shell:
    //   read _bas[shell] → atom_id, l, nprim, nctr, ptr_exp, ptr_coeff
    //   read _atm[atom_id] → ptr_coord
    //   read env[ptr_coord..ptr_coord+3] → atom xyz
    //   read coords[g, :] → grid xyz
    //   r² = (gx - ax)² + ...; r = sqrt(r²)
    //   For each primitive p ∈ 0..nprim:
    //     read env[ptr_exp + p] → α_p
    //     gauss_p = exp(-α_p * r²)
    //     For each contraction c ∈ 0..nctr:
    //       coeff = env[ptr_coeff + c*nprim + p]  (F-order coeff matrix)
    //       contracted_radial[c] += coeff * gauss_p
    //   For each contraction c, for each ao within the (l,c) shell-block:
    //     apply cart2sph(l) transform if spherical
    //     write to out[g + ao*ngrids]  (F-order: ao_idx outer, grid inner)
}
```

**One-launch vs chunked (Claude's discretion):**
- Cubecl's CPU runtime handles arbitrary work-group counts; chunking is needed when device memory < input size, which for `ngrids ≤ 100k × nao ≤ 1000 × 8 bytes = 800 MB` rarely binds for HF/DFT/MP2/CCSD molecules. **Recommendation: one launch per `eval_gto` call; revisit chunking in Phase 4 (DFT) if grid-size benchmarks show device-mem pressure.** This matches the cubecl-matmul example pattern (`docs/manual/Cubecl/cubecl_matmul_gemm_example.md`) which is also one-launch.
- `BLKSIZE = 56` upstream (in `eval_gto.py:26`) is the C-level grid-block size for the libcgto-driver-call, not directly applicable to a cubecl kernel; it informs the work-group dimension if planner chooses to chunk.

**Algebra-wall compliance (D-04 strict):** `pyscf-gto::eval_gto` must NOT name `cubecl::Array`, `CubeCount`, `CubeDim`, or any `cubecl::*` type. It calls a launch helper exported from `pyscf-kernels` like:

```rust
// pyscf-kernels exports — only API pyscf-gto sees:
pub fn eval_gto_sph(
    client: &pyscf_algebra::AlgebraClient,
    coords: &pyscf_algebra::Tensor,
    atm: &pyscf_algebra::Tensor,
    bas: &pyscf_algebra::Tensor,
    env: &pyscf_algebra::Tensor,
    ao_loc: &pyscf_algebra::Tensor,
    out: &mut pyscf_algebra::Tensor,
) -> Result<(), pyscf_algebra::AlgebraError>;
```

Inside `pyscf-kernels::eval_gto_sph`, the function matches on `client.kind()` (existing `AlgebraClient` enum from Phase 1) and dispatches to the cubecl runtime arm — same pattern Phase 1 established for `gemm`/`reduce_sum` (`crates/pyscf-algebra/src/client.rs:10-18`).

### Pattern 5: `set_geom_` granular cache invalidation (GTO-10)

**What:** When the user mutates geometry (`mol.set_geom_("H 0 0 0; H 0 0 1.5")`), basis structure is preserved — only atom coordinates change. Re-running `make_env` from scratch is correct but throws away the parsed-basis cache.

**Granular invalidation:**
```rust
// pyscf-gto/src/format_atom.rs
impl Mole {
    pub fn set_geom_(&mut self, new_atom: &str) -> Result<&mut Self, PyscfRsError> {
        let parsed = format_atom(new_atom, self.unit, self.origin, self.axes)?;
        if parsed.len() != self._atom.len() {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                format!("set_geom_ atom count mismatch: was {}, got {}", self._atom.len(), parsed.len())
            )));
        }
        // Validate symbol-by-symbol — set_geom_ must NOT change which atoms are which.
        for (old, new) in self._atom.iter().zip(&parsed) {
            if old.0 != new.0 {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                    format!("set_geom_ atom symbol mismatch at index: was {}, got {}", old.0, new.0)
                )));
            }
        }
        self._atom = parsed;
        // _atm rows: PTR_COORD slots stay the same (env slots), but _env coordinate values update.
        for (i, (_sym, coord)) in self._atom.iter().enumerate() {
            let row_start = i * 6;
            let ptr_coord = self._atm[row_start + cintx_compat::raw::PTR_COORD] as usize;
            self._env[ptr_coord]     = coord[0];
            self._env[ptr_coord + 1] = coord[1];
            self._env[ptr_coord + 2] = coord[2];
        }
        // _bas, _basis, ALIAS-resolved data, ao_loc_nr, nao_nr ALL preserved unchanged.
        // (atom_charges, nelectron, etc. unchanged because charges didn't change.)
        Ok(self)
    }
}
```

This invalidation is fast (O(natm)) and correct because `_env`'s only geometry-dependent slots are the per-atom xyz coordinates — basis exponents and coefficients live elsewhere in `_env` and are independent of geometry.

### Anti-Patterns to Avoid

- **Hand-defining libcint slot constants** in pyscf-gto. Reuse `cintx_compat::raw::*` per D-03. Drift between cintx and pyscf-rs slot definitions silently breaks byte-identity. Phase 2 PR review must include: "did this PR import any constant from `cintx_compat::raw`, and if so, did it redefine it?"
- **Embedding the basis files at build time** via `include_bytes!` or `build.rs` parsing. D-01 forbids it. The first reason is "don't freeze compile" (a project-wide preference recorded in user memory + 02-CONTEXT.md `<specifics>`); the second reason is that the wheel-packaging story in Phase 8 already plans to bundle these files separately.
- **Hand-rolling spherical-harmonic transforms in pyscf-gto.** The transform tables live in `pyscf/gto/mole.py:157-258` (`cart2sph`, `cart2spinor_kappa`, etc.). Phase 2 ports those tables into a `pyscf-gto::sph_harmonics` module — they are small static matrices, not algorithms to invent.
- **Returning C-order arrays where upstream returns F-order** (Pitfall 8). See "Common Pitfalls" §1.
- **Skipping the per-symbol basis grouping** in `make_env` step 3. Upstream groups by element symbol and clones the basis template per atom. A naive "build basis per atom from scratch" produces the same atom-coordinate `_atm` rows but a DIFFERENT `_env` byte layout (because per-atom data is concatenated in atom-order, but per-symbol basis data is concatenated in first-occurrence-order). The byte-identity test catches this; planner must respect the upstream order.

### Anti-Pattern: `pyscf-gto` importing `cubecl-*`
The algebra wall (D-04 + Phase 1 D-04..D-06) forbids it. Enforced by xtask lint per Phase 1 Plan 05. Any `cubecl::Array`, `CubeCount`, `CubeDim`, `client.create(...)` reference inside `pyscf-gto/src/**` is a CI-blocking failure.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---|---|---|---|
| Workspace planning + chunk scheduling for 1e/2e integrals | A pyscf-gto-side workspace allocator | `cintx_rs::SessionRequest::query_workspace().evaluate()` | cintx already implements this with libcint policy gates, memory-limit pre-flight, and chunk scheduling (`cintx-rs/src/api.rs:62-285`). Reinventing duplicates the most subtle part of the integral facade. |
| Boys function evaluation for 2e integrals | Boys-function table (Pitfall 18) | Inside cintx (delegated) | Boys function accuracy is one of the hardest numerical problems in chem software. Pitfall 18 explicitly delegates to cintx. **Phase 2 should never compute Boys values** — if it appears, redirect to cintx. |
| Basis-set parsing of NWChem / CP2K text | A regex-based ad-hoc parser | Port `pyscf/gto/basis/parse_nwchem.py` (and the 3 siblings) function-for-function | The parsers handle subtle cases (multiple-element blocks, comment lines, BASIS SET delimiters, integer-l labels like "S P D"). A from-scratch parser is a guaranteed bug source. |
| Spherical-harmonic transforms (`cart2sph_l`) | Hand-derived transform matrices | Port `cart2sph` / `cart2spinor_kappa` tables from `pyscf/gto/mole.py:157-258` | These are explicit integer-coefficient matrices from the literature. Re-deriving them is error-prone (sign conventions, normalization). |
| GEMM / reduce / axpy inside the eval_gto kernel | Hand-rolled SIMD or `std::simd` | `pyscf-algebra` primitives (Phase 1 D-06) | Algebra wall enforced; Phase 1 already shipped this surface. |
| Permutation-symmetry exploitation in 2e integrals | A custom `aosym='s4'`-style fold | cintx already supports this via `ExecutionOptions` | `IntegralTensor.extents` carries the symmetry; pyscf-gto only translates the upstream `aosym='s1'`/`'s4'` flag to cintx's `ExecutionOptions`. |
| HDF5 round-trip of basis state for `mol.dumps()` / `loads()` | Hand-rolled JSON | `serde_json` with a `MoleSerializable` snapshot struct | GTO-09 only requires semantic round-trip, not byte-identical to upstream JSON. `serde_json` is in workspace deps. |
| Per-atom env-buffer sizing | Manual `Vec::with_capacity` math | Use the same algorithm as upstream `make_env` — it walks the structure twice and `_env` ends at `ptr_env` after the second pass | Upstream's two-pass approach handles ECP, finite-nucleus, GHOST atoms uniformly. A "compute total size up front" optimisation introduces drift risk for byte-identity. |

**Key insight:** Phase 2 is a glue phase. The hard numerical work lives in cintx (integrals), upstream PySCF (algorithms — Apache-2.0 port-friendly), and the tested cubecl primitives Phase 1 shipped. The pyscf-rs invention is the typed rust API and the basis-loader cache; everything else is delegation.

---

## Runtime State Inventory

> Phase 2 is greenfield (replaces Phase 1 stubs with full implementations). No rename / refactor of stored runtime state. The only "state" that Phase 2 introduces and downstream phases inherit:

| Category | Items Found | Action Required |
|---|---|---|
| Stored data | None — Phase 2 ships zero datastores. The `OnceLock<HashMap<String, ParsedBasis>>` cache is **process-local** (per-process, fresh on each `cargo run`/`pytest` start), no persistence. | None |
| Live service config | None — no external services Phase 2 depends on. cintx is a sibling crate (path dep), not a service. | None |
| OS-registered state | None — Phase 2 ships no daemons, scheduled tasks, or OS hooks. | None |
| Secrets/env vars | **NEW: `PYSCF_BASIS_PATH`** (D-02) — analogous to Phase 1's `PYSCF_BACKEND` / `PYSCF_DTYPE`. Read at first basis-load via `std::env::var`. Documented in `CONTRIBUTING.md` as a Phase 2 deliverable (analog to Phase 1's `docs/upgrade-cubecl.md`). | Document the env var; add to a `docs/env-vars.md` if planner introduces one (or extend Phase 1's CONTRIBUTING.md section). Phase 3 PyO3 wrapper exposes this as `pyscf.__config__.PYSCF_BASIS_PATH` for parity. |
| Build artifacts | None — D-01 forbids `build.rs` codegen. The `OnceLock` cache is fully runtime. | None |

**The canonical question — answered:** *After every file in the repo is updated, what runtime systems still have the old state cached, stored, or registered?* **Nothing.** Phase 2 is process-local + read-only against `pyscf/gto/basis/`. The only deliverable runtime state is the new `PYSCF_BASIS_PATH` env var, which is opt-in.

---

## Common Pitfalls

### Pitfall 1: F-order vs C-order array layout (Pitfall 8 from ROADMAP)
**What goes wrong:** Upstream `mol.intor("int2e")` returns a 4D ndarray in **F-order** (Fortran column-major); `mol.intor("int1e_ovlp_sph")` is also F-order. But Rust `Vec<f64>` and naive C-style indexing assume row-major. If pyscf-rs returns a row-major buffer where upstream returns column-major, the byte-identity test fails AND every downstream method (SCF Fock build, MP2 AO→MO transform, CCSD T2 update) consumes the wrong layout.

**Why it happens:** cintx's `IntegralTensor.owned_values` is a flat `Vec<f64>` with explicit `extents: Vec<usize>` and a `component_axis_leading: bool` flag (`cintx-rs/src/api.rs:439-445`). The flag indicates whether the COMPONENT axis is leading (e.g., the "3" in `int1e_ipnuc` for ∇x/∇y/∇z), not the AO axes. The layout of the AO axes (i, j or i, j, k, l) is by convention F-order.

**How to avoid:**
1. For each integral name, look up the upstream layout in `pyscf/gto/moleintor.py` (the `_INTOR_FUNCTIONS` dict at line 288+ has the component count; the array shape comes from the calling site in upstream — `getints2c`/`getints4c` return `(comp, naoi, naoj)` etc. with implicit F-order via `numpy.ndarray(shape, order='F', buffer=...)`).
2. Translate cintx's `extents` + `owned_values` to the upstream layout in pyscf-gto's `intor.rs`. The translation is index-pattern math — no data copy unless permutation is needed (and for many integrals, F-order matches the natural cintx output and a `Vec` re-interpretation suffices).
3. Document the chosen output layout per intor name in a comment block at the dispatch site. **Don't** assume "all integrals are F-order"; some derivative families (`int1e_ipovlp` etc.) have leading component axis.

**Warning signs:** byte-identity test passes for `int1e_ovlp_sph` (2D, sym, F-order indistinguishable from C-order at low ang-mom) but fails for `int1e_ipnuc_sph` (3D, component-leading) or `int2e` (4D).

### Pitfall 2: Off-by-one in basis indexing (Pitfall 17 from ROADMAP)
**What goes wrong:** `_bas[ATOM_OF]` is 0-based in libcint convention but upstream PySCF user-facing methods (`mol.bas_atom(i)`, `mol.atom_charge(i)`) are also 0-based. ECP `_atm[NUC_MOD_OF]=2` switches the atom to ECP-mode and **subtracts** core electrons from `CHARGE_OF` — this happens INSIDE `make_ecp_env` (`mole.py:1149`). Forgetting that step, or applying it twice, breaks `tot_electrons` and every SCF energy.

**Why it happens:** The slot semantics are subtle. `CHARGE_OF` literally stores `charge - n_core_electrons` for ECP atoms; `NUC_MOD_OF` stores the model code. A naive "compute total charge from `_atm[CHARGE_OF].sum()`" does the right thing only AFTER ECP processing.

**How to avoid:**
1. Treat `make_ecp_env` as the **only** writer of ECP-related slots. Run it after `make_env` (atom + basis), as upstream does.
2. Audit `tot_electrons` (`mole.py:1162-1186`) — it sums `atom_charges()` (which reads `_atm[:, CHARGE_OF]`), THEN subtracts user-set `mol.charge`. The "subtract core electrons" step happens inside the `_atm` projection, not at total-electron computation time.
3. Assert `_atm.len() == natm * ATM_SLOTS` and `_bas.len() == nbas * BAS_SLOTS` at the end of `build()`. These are cheap and catch slot-width mistakes.

**Warning signs:** `mol.nelectron` is off by an even number (= 2× per ECP atom's missing core), or HF SCF energy drifts by exactly the core-orbital energies.

### Pitfall 3: Boys-function accuracy (Pitfall 18 from ROADMAP — DELEGATED)
**What goes wrong:** Boys function `F_n(T)` evaluation is required for 2e Coulomb integrals over Gaussians. Naive series expansion loses precision in the medium-T regime; downhill recursion is unstable for high n. Hand-rolling produces "approximately right" integrals that drift at the µHartree-bit level.

**How to avoid:** **Don't compute Boys in pyscf-rs.** It lives in cintx (it's an integral-evaluation concern). Phase 2 calls `mol.intor("int2e_sph")` → cintx-rs → cintx-cubecl → libcint-equivalent kernel → Boys evaluation. If a Boys reference appears in pyscf-gto code review, redirect it to a cintx issue.

**Warning signs:** Reviewer should grep `pyscf-gto/src/**` for `boys`, `gamma_inc`, `erf` after every PR.

### Pitfall 4: Basis grouping by symbol vs by atom (`make_env` step 3)
**What goes wrong:** As noted in Pattern 3 / Anti-pattern 5: upstream `make_env` builds the per-element basis template ONCE, then clones for each atom of that element. A "build per atom" reorder produces different `_env` byte layout. The arrays still describe the same molecule mathematically, but byte-identity (Phase 2 success criterion 1) fails.

**How to avoid:** Mirror upstream's two-pass loop literally:
- Pass 1 over atoms: `make_atm_env`, append to `_atm`, append xyz to `_env`.
- Group basis by **first-occurrence symbol order**: iterate `for symb, basis_for_symb in basis.items()` (Python dict insertion-order, post-Python 3.7). In Rust, use `IndexMap` or sort by first-occurrence index.
- Pass 2 over atoms: clone the per-symbol template, patch `ATOM_OF`, append to `_bas`.

**Warning signs:** byte-identity assertion fails on the `_env` array specifically; `_atm` and `_bas[:, all-but-ATOM_OF]` match.

### Pitfall 5: `cart=True` / `cart=False` inconsistency between `nao_nr` and `_bas[KAPPA_OF]`
**What goes wrong:** PySCF's libcint convention: `KAPPA_OF=0` means "use the integer-l (sph or cart) code path"; nonzero kappa = spinor (4-component). For `mol.cart=True`, kappa stays 0 — the cart-vs-sph choice happens at intor-name suffix time (`int1e_ovlp_sph` vs `int1e_ovlp_cart`). For `mol.cart=False`, also kappa=0.

**Why it happens:** A reasonable-looking implementation might encode "cart vs sph" into KAPPA_OF, breaking byte-identity when cintx then ignores the field.

**How to avoid:**
1. **Always set `_bas[i, KAPPA_OF] = 0`** unless the spinor scope expands (out of v1).
2. The `mol.cart` flag is consumed at TWO sites: (a) `nao_nr` returns `nao_cart` if `cart` else `2l+1` sum (`mole.py:1378-1389`); (b) `mol.intor(name)` appends `_cart`/`_sph` suffix at dispatch — `_add_suffix` at `mole.py:945+`.
3. Add a unit test: `mol.cart=True` produces `_bas[:, KAPPA_OF].iter().all(|&k| k == 0)`.

**Warning signs:** `nao_nr` differs from upstream when `cart=True`; or `mol.intor` produces wrong-shape arrays.

### Pitfall 6: Lazy-init race in basis loader
**What goes wrong:** `OnceLock<HashMap<String, ParsedBasis>>` is thread-safe, but if Phase 3 PyO3 introduces multi-threaded calls into `mol.build()`, two threads might race on first-load — only one wins, but the loser observes a stale `Err` if the loader path resolution panicked.

**How to avoid:**
1. Use `OnceLock::get_or_try_init` (stable in Rust 1.95+ via `OnceLock::get_or_init` returning `&T` after panic recovery — but `get_or_try_init` is gated). Actually, **prefer `OnceLock<Result<HashMap<...>, BasisLoadError>>`** so the error is cached too. Or wrap each basis lookup in its own `OnceLock` (per-name) so a single missing file doesn't poison the whole cache.
2. **Recommendation: per-name `OnceLock<Result<ParsedBasis, BasisLoadError>>` inside an outer `RwLock<HashMap<String, OnceLock<Result<...>>>>`.** Verbose, but the only safe shape for "cache failures, retry success".
3. Or simpler: `Mutex<HashMap<String, ParsedBasis>>`. Acceptable for Phase 2 because basis-load is dwarfed by integral evaluation. Optimize later if a profiler shows it.

**Warning signs:** "Tests pass in isolation, fail under `cargo test` parallel runner" pattern.

---

## Code Examples

Verified patterns (Apache-2.0 port from upstream PySCF, in this repo):

### Example 1: `pyscf.M(...)` factory (port of `mole.py:106-118`)

```rust
// crates/pyscf-gto/src/lib.rs

/// Shortcut to build a Mole. Equivalent to `pyscf.M(...)` upstream.
///
/// Source: pyscf/gto/mole.py:106-118 (Apache-2.0)
pub fn M(args: MoleBuildArgs) -> Result<pyscf_core::Mole, PyscfRsError> {
    let mut mol = pyscf_core::Mole::default();
    mol.build_from(args)?;
    Ok(mol)
}
```

`MoleBuildArgs` is the typed kwargs analog: `{ atom: AtomInput, basis: BasisInput, charge: i32, spin: i32, ecp: Option<EcpInput>, unit: Unit, verbose: u8, ... }`. Phase 3 PyO3 wraps this with `#[pyfunction]` taking Python kwargs.

### Example 2: `format_atom` for the string form (port of `mole.py:320-415`)

```rust
// crates/pyscf-gto/src/format_atom.rs

/// Parse the string form of `atom`. Mirrors pyscf/gto/mole.py:373-392.
fn parse_atom_string(s: &str, unit: Unit, origin: [f64; 3], axes: [[f64; 3]; 3])
    -> Result<Vec<(String, [f64; 3])>, PyscfRsError>
{
    // First, check if it's a file path.
    if std::path::Path::new(s).is_file() {
        return parse_atom_file(s, unit, origin, axes);
    }

    // Normalize separators: ";" → "\n", "," → " ", "\t" → " "
    let normalized = s.replace(';', "\n").replace(',', " ").replace('\t', " ");

    let mut atoms = Vec::new();
    for line in normalized.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // "Symbol x y z" — 4 whitespace-separated tokens
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.len() < 4 {
            // Z-matrix form — defer to from_zmatrix port (Phase 2 stub: NotYetImplemented)
            return Err(PyscfRsError::NotYetImplemented {
                phase: 2, what: "Z-matrix atom form (defer to plan)",
            });
        }
        let symb = atom_symbol(tokens[0])?;          // Apply _atom_symbol parsing
        let xyz: [f64; 3] = [
            tokens[1].parse().map_err(|_| invalid_coord(line))?,
            tokens[2].parse().map_err(|_| invalid_coord(line))?,
            tokens[3].parse().map_err(|_| invalid_coord(line))?,
        ];
        atoms.push((symb, xyz));
    }

    // Apply unit conversion + origin shift + axes rotation
    let unit_factor = unit.length_in_au();
    let mut out = Vec::with_capacity(atoms.len());
    for (symb, xyz) in atoms {
        let shifted = [xyz[0] - origin[0], xyz[1] - origin[1], xyz[2] - origin[2]];
        let rotated = [
            unit_factor * (axes[0][0]*shifted[0] + axes[1][0]*shifted[1] + axes[2][0]*shifted[2]),
            unit_factor * (axes[0][1]*shifted[0] + axes[1][1]*shifted[1] + axes[2][1]*shifted[2]),
            unit_factor * (axes[0][2]*shifted[0] + axes[1][2]*shifted[1] + axes[2][2]*shifted[2]),
        ];
        out.push((symb, rotated));
    }
    Ok(out)
}
```

### Example 3: `make_atm_env` (port of `mole.py:961-980`, uses cintx-compat constants per D-03)

```rust
// crates/pyscf-gto/src/make_env.rs

use cintx_compat::raw::{ATM_SLOTS, CHARGE_OF, PTR_COORD, NUC_MOD_OF};

const PTR_ZETA: usize = 3;       // libcint convention; not in cintx-compat — confirmed in upstream mole.py
const NUC_POINT: i32 = 0;
const NUC_GAUSS: i32 = 1;
const NUC_ECP:   i32 = 2;

/// Append (charge, ptr_coord, nuc_mod) to _atm; xyz to _env. Returns updated ptr_env.
/// Source: pyscf/gto/mole.py:961-980 (Apache-2.0)
fn make_atm_env(
    atom: &(String, [f64; 3]),
    nuc_charge: i32,
    nuclear_model: i32,
    ptr_env: usize,
    _env: &mut Vec<f64>,
    _atm: &mut Vec<i32>,
) -> usize {
    let zeta: f64 = match nuclear_model {
        m if m == NUC_POINT => 0.0,
        m if m == NUC_GAUSS => dyall_nuc_mod(nuc_charge),  // port of mole.py helper
        _ => 0.0,
    };
    _env.extend_from_slice(&atom.1);    // 3 doubles: x, y, z
    _env.push(zeta);                    // 1 double: zeta

    let row_start = _atm.len();
    _atm.resize(row_start + ATM_SLOTS, 0);
    _atm[row_start + CHARGE_OF]  = nuc_charge;
    _atm[row_start + PTR_COORD]  = ptr_env as i32;
    _atm[row_start + NUC_MOD_OF] = nuclear_model;
    _atm[row_start + PTR_ZETA]   = (ptr_env + 3) as i32;
    // slots 4, 5 stay zero (matches upstream behavior)

    ptr_env + 4    // moved 4 doubles forward (3 xyz + 1 zeta)
}
```

### Example 4: `mol.intor` dispatch (Phase 2 invention)

```rust
// crates/pyscf-gto/src/intor.rs

use cintx_rs::{SessionRequest, ExecutionOptions};
use cintx_core::{OperatorId, Representation, ShellTuple};

impl pyscf_core::Mole {
    /// Compute the named integral. Thin wrapper over cintx for in-scope intors.
    /// ECP intors (int1e_ecp*) route to EcpEngine per D-07.
    pub fn intor(&self, name: &str) -> Result<IntegralOutput, PyscfRsError> {
        // Suffix normalization: append _sph or _cart per mol.cart per pyscf/gto/mole.py:945+
        let full_name = self.add_suffix(name);

        // ECP routing per D-07
        if full_name.starts_with("int1e_ecp") || full_name.starts_with("ECPscalar") {
            let engine = self.ecp_engine();    // Returns &dyn EcpEngine
            return engine.ecp_int1e(self, &full_name);
        }

        // Resolve name → cintx OperatorId via the lookup table
        let operator = resolve_intor_name(&full_name)
            .ok_or_else(|| PyscfRsError::Core(CoreError::InvalidMolecule(
                format!("unknown intor: {}", full_name)
            )))?;

        let representation = if self.cart { Representation::Cart } else { Representation::Spheric };
        let basis = self.cintx_basis();    // Arc<BasisSet>, zero-copy per GTO-11
        let shells = basis.shell_tuple_for_indices(0..self.nbas).map_err(...)?;
        let options = ExecutionOptions::default();    // Phase 2 doesn't tune; defaults are fine

        let request = SessionRequest::new(operator, representation, &basis, shells, options);
        let query = request.query_workspace().map_err(...)?;
        let output = query.evaluate().map_err(...)?;

        // output.tensor.owned_values is Vec<f64>; output.tensor.extents is Vec<usize>.
        // Translate to upstream PySCF layout (F-order for most names — see Pitfall 1).
        translate_intor_output(&full_name, output.tensor)
    }
}
```

The `resolve_intor_name` function is the only Phase 2 invention; it's a static `match` table from intor name (e.g., `"int1e_ovlp_sph"`) to `cintx_core::OperatorId`. Source-of-truth for the catalogue is `cintx/crates/cintx-compat/src/raw.rs` `RawApiId` enum (referenced in CONTEXT.md canonical_refs).

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|---|---|---|---|
| Per-atom rebuild of basis-data block in `_env` | Per-symbol grouping then per-atom template-clone (Pattern 3 step 3) | Upstream PySCF since `mole.py` inception (2014) | Required for byte-identity; new entrants commonly miss this |
| `include_bytes!` of basis-set files at build time | Runtime `OnceLock` + path resolver (D-01 + D-02) | Project decision 2026-05-10 (CONTEXT.md) | Avoids "freeze compile" pitfall; matches the wheel-bundling story for Phase 8 |
| `int1e_ecp` evaluation inside pyscf | ECP belongs in cintx; pyscf-rs ships only loading + trait shim (D-05 + D-06) | Project decision 2026-05-10 | Cleanly partitions concerns; gap-closure plan handles cintx ECP landing |
| Hand-coded `_atm`/`_bas` slot constants per crate | Single source in `cintx-compat::raw` (D-03) | Project decision 2026-05-10 | Eliminates drift between cintx + pyscf-rs |

**Deprecated/outdated:**
- Upstream PySCF's `from_zmatrix` Z-matrix parser is included in `format_atom` but rare in modern use. Phase 2 ships a stub returning `NotYetImplemented` if user supplies < 4 tokens per atom-line; full Z-matrix is a Phase 2.x or Phase 3 enhancement. (Listed as discretion.)
- Upstream's `_load_external` BSE (Basis Set Exchange) network fetch (`pyscf/gto/basis/__init__.py:699-712`) is **out of scope** for Phase 2. We don't ship a network basis fetch; users either use the 184 builtin files or supply their own NWChem/CP2K text. Documented in plan.
- Upstream's `parse_molpro.py`, `parse_gaussian.py`, `parse_bfd_pp.py` are smaller parsers not in the canonical 4-set. Phase 2 may leave these as `NotYetImplemented { phase: 2.x }` if not needed for the test corpus.

---

## Assumptions Log

> Claims tagged `[ASSUMED]` requiring user / planner confirmation before commit:

| # | Claim | Section | Risk if Wrong |
|---|---|---|---|
| A1 | "184 unique basis files" excludes 5 subdirectories' contents (~120 more files via `find -name '*.dat'` would yield 305 total). The "207 in-scope" figure in CONTEXT.md is a project-internal estimate that doesn't match either count exactly. | Standard Stack table (`Upstream pyscf/gto/basis/`) | Test corpus tier sizing — the small PR-CI corpus is ~5 representative basis sets; the nightly sweep (Phase 8 ORACLE-06) needs the exact count. The "all 207 resolve" success criterion in GTO-03 should be re-stated as "all builtin basis files present in the runtime path resolve" — which is correct regardless of N. **Recommendation:** plan should accept "all builtin basis files" as the contract, drop the literal 207. |
| A2 | The `gto_norm` + `_nomalize_contracted_ao` upstream defaults (`NORMALIZE_GTO=True`) produce a specific coefficient transform; the `gaussian_int(2l+2, 2*expnt)` integral has a closed form via `scipy.special.gamma` that maps to a simple Rust expression. | Pattern 3 (Coefficient normalisation note) | If the closed-form mapping is wrong (e.g., off by a factor of 2 in `expnt`), every integral byte-identity check fails. Mitigation: cross-check with `pyscf/gto/mole.py:120-155` `gaussian_int` expression and verify against a known basis (e.g., He/STO-3G). |
| A3 | cintx's `Representation::{Cart, Spheric, Spinor}` enum (`cintx-core::operator::Representation`, referenced from `cintx-rs/src/api.rs:6`) maps 1:1 to PySCF's `_sph`/`_cart`/`_spinor` intor-name suffixes. | Code Examples §4 (`mol.intor` dispatch) | If the mapping is reversed or off, dispatch picks the wrong cintx operator; integrals come out transposed. Mitigation: Wave 0 smoke test asserts `int1e_ovlp_sph` for H2 STO-3G matches the analytical 2×2 result. |
| A4 | The per-name `_INTOR_FUNCTIONS` table at `pyscf/gto/moleintor.py:288+` is the source of truth for "comp" (component count) and the table covers every in-scope intor for SCF/DFT/MP2/CCSD/grad. The Phase 2 dispatcher needs this table verbatim. | Code Examples §4 | If we miss intors used by Phase 3+ (e.g., `int1e_grids` for solvent, `int1e_ipovlp` for grad), the SCF/grad phases hit `unknown intor` errors. Mitigation: oracle test diff the `_INTOR_FUNCTIONS` keys against the in-scope-method intor calls upstream — produce a Phase 2 catalogue, plan a coverage assertion. |
| A5 | `pyscf-kernels` declaring a direct `cubecl` dependency satisfies the algebra wall (D-04 says pyscf-gto and pyscf-kernels both go through pyscf-algebra; but a kernel implementation crate inherently must touch cubecl). The wall is enforced on consumer crates, not on `pyscf-algebra` itself. | Architecture Patterns §"Crates that get NEW dependencies" | If the algebra-wall lint (xtask check from Plan 01-05) treats `pyscf-kernels` as a "method crate" forbidden from `cubecl-*`, Phase 2 build fails. **Mitigation:** verify the lint allowlist at `xtask/src/check_algebra_wall.rs` (or wherever Plan 01-05 implemented it). If `pyscf-kernels` is NOT on the allowlist, planner adds it as a Wave 0 task (small, surgical). |
| A6 | `cintx-compat` is reachable from `pyscf-gto` (currently only `pyscf-algebra` and `pyscf-runtime` declare cintx deps in the workspace tree based on the path-dep patch). Adding `cintx-compat = "*"` to `pyscf-gto/Cargo.toml` resolves correctly via the existing `[patch.crates-io]` cintx redirect. | Architecture Patterns §"Crates that get NEW dependencies" | If the cintx workspace doesn't publish `cintx-compat` separately on crates.io, the patch redirect fails (the patch substitutes a registry crate, not adds a path crate). **Mitigation:** verify by `cargo metadata --format-version 1` in cintx workspace — confirm `cintx-compat` is its own package. The cintx file path `cintx/crates/cintx-compat/` confirms it's a workspace member; the question is whether it's published. If unpublished, add a `[patch.crates-io] cintx-compat = { path = "../cintx/crates/cintx-compat" }` entry to root Cargo.toml. |
| A7 | The `eval_gto_sph` cubecl kernel sketched in Pattern 4 fits cubecl 0.10.0's `#[cube(launch_unchecked)]` macro surface (`Array<f64>`, `comptime` u32 args, `ABSOLUTE_POS`). cubecl-matmul/-reduce are NOT needed for eval_gto. | Architecture Patterns §Pattern 4 | If cubecl 0.10.0's actual macro surface differs (e.g., `Tensor` instead of `Array` for backed handles), the kernel won't compile. **Mitigation:** Wave 0 includes a "cubecl-cpu vector_add minimal kernel" smoke test in `pyscf-kernels` that proves the launch shape works. Cintx-cubecl/xcfun-kernels precedent crates already use cubecl 0.10.0 — copy their `Cargo.toml` + a one-shot kernel pattern. |
| A8 | `mol.cart` flag does NOT change `_bas[:, KAPPA_OF]` (always 0 for in-scope work); cart vs sph is a name-suffix-time decision per Pitfall 5. | Pitfall 5 | If upstream secretly encodes cart-vs-sph elsewhere (e.g., in a hidden meta-array), pyscf-rs ships wrong arrays under `mol.cart=True`. **Mitigation:** unit test sets `mol.cart=True` on H2/cc-pVDZ and asserts `_bas` byte-identity to upstream's `mol.cart=True` reference. |

---

## Open Questions

1. **Does the `cintx_rs::IntegralTensor.owned_values` layout match upstream `numpy.ndarray(shape, order='F')` for in-scope intors, or do certain intors ship in C-order?**
   - What we know: `IntegralTensor.extents` carries explicit shape; `component_axis_leading` distinguishes leading-component arrays.
   - What's unclear: upstream's per-intor F/C order convention is documented per call-site in `getints2c`/`getints4c` (`moleintor.py:475+` / `:603+`) but not in a single table.
   - Recommendation: Wave 0 deliverable — build a per-intor layout table by running upstream `mol.intor(name).flags['F_CONTIGUOUS']` for each name in scope; commit to `pyscf-gto/docs/intor_layouts.md`. This is concrete enough to be a Plan task.

2. **Does `pyscf-kernels` qualify for the algebra-wall lint allowlist alongside `pyscf-algebra` and `pyscf-runtime`?**
   - What we know: D-04 puts the eval_gto cubecl kernel in `pyscf-kernels`. The kernel inherently touches `cubecl::*` types.
   - What's unclear: whether Plan 01-05's lint distinguishes "kernel impl crates may import cubecl" from "method crates may not".
   - Recommendation: Read `xtask/src/check_algebra_wall.rs` (or wherever the lint lives) at Wave 0. If pyscf-kernels is excluded from the wall, no action; if included, add a one-line allowlist entry.

3. **`mol.dumps()` JSON wire format — match upstream literally or adopt a Rust-native shape?**
   - What we know: GTO-09 + Claude's discretion permits "semantic round-trip, not byte-identical to upstream JSON". CONTEXT.md endorses this.
   - What's unclear: Phase 3 PyO3 binding may force upstream-JSON compatibility for chkfile interop (ORACLE-08).
   - Recommendation: Phase 2 ships `serde_json` semantic round-trip; Phase 3 adds a `mol.dumps_pyscf_compat()` method if needed for chkfile interop. **Action item for planner: surface this as a known gap in the Phase 3 PLAN.md.**

4. **NORMALIZE_GTO default behavior — Phase 2 mirrors upstream's `True` default, but does any intor regression test set `False`?**
   - What we know: `NORMALIZE_GTO=True` is the upstream default at `mole.py:1004-1006`.
   - What's unclear: rare paths (decontracted basis, custom-basis tests) may set `False`.
   - Recommendation: Phase 2 default = `True`; add a `mol.normalize_gto: bool` attribute (preserves upstream surface) but only test with `True` until a counterexample appears. Document in plan.

5. **`USER_BASIS_DIR` / `USER_BASIS_ALIAS` semantics — env var only, programmatic API, or both?**
   - What we know: upstream uses `pyscf.__config__.USER_BASIS_DIR` (module-import-time).
   - What's unclear: which surface Phase 3 PyO3 will adopt for `pyscf.__config__` (it's a Python config-dict pattern, doesn't translate cleanly).
   - Recommendation: Phase 2 ships `PYSCF_USER_BASIS_DIR` env var (mirrors `PYSCF_BASIS_PATH`); Phase 3 PyO3 adds the programmatic API as a `pyscf.__config__` shim object.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|---|---|---|---|---|
| `cargo` | Build, test | ✓ | 1.95.0 (`f2d3ce0bd 2026-03-21`) | — |
| `rustc` | Build | ✓ | 1.95.0 (`59807616e 2026-04-14`) | — |
| `python3` | Oracle harness (Phase 2 byte-identity tests against upstream PySCF) | ✓ | 3.14.4 | — |
| Sibling `cintx` workspace | `cintx-core::BasisSet`, `cintx-compat::raw`, `cintx-rs::SessionRequest` (D-03, GTO-06, GTO-11) | ✓ | path: `/home/user/Documents/workspace/cintx` (already in `[patch.crates-io]` Cargo.toml:93) | — |
| Sibling `xcfun_rs` workspace | Phase 4 only (DFT) — not needed in Phase 2 | ✓ | path-dep | — |
| Sibling `libxc_rs` workspace | Phase 4 only (DFT-03) — **DISABLED in Cargo.toml:94** (libxc compile is ~6h per user memory) | ✗ (intentionally) | — | Phase 2 does NOT need libxc_rs. Re-enable in Phase 4. |
| `pyscf` Python package (upstream, in this repo) | Oracle harness (compare _atm/_bas/_env byte-for-byte) | ✓ | checked into repo at `pyscf/` | — |
| Cargo.lock | Reproducible build | ✓ | exists at repo root | — |
| `cubecl-cpu` (default backend) | `pyscf-kernels::eval_gto` Phase 2 MVP | depends on Plan 01-04 outcome | (workspace pin `=0.10.0`) | If cubecl-cpu can't build for any reason, eval_gto can ship a host-only fallback (compute in Vec<f64>, no kernel launch) for Phase 2; revisit in Phase 4 when DFT puts pressure on the GPU paths. |
| Existing `pyscf-algebra` Phase 1 surface (`AlgebraClient`, `Tensor`, `select_backend`) | `pyscf-kernels::eval_gto` launch | ✓ | non-stub since Plan 01-04 | — |
| Existing Phase 1 `pyscf-core` stubs (`Mole`, `BasisSet`, `IntegralEngine`) | Phase 2 fills bodies | ✓ | non-stub since Plan 01-02 (`crates/pyscf-core/src/{mole,basis_set,traits,error}.rs`) | — |

**Missing dependencies with no fallback:** None.

**Missing dependencies with fallback:** None blocking.

**Note (Phase 1 BLOCKER carry-over from STATE.md):** Phase 1 has 2 unmerged gap-closure plans (`01-08-PLAN.md` cintx clean-SHA repin, `01-09-PLAN.md` check-cubecl-pin). The local-path-dep workaround in Cargo.toml:92-95 keeps Phase 2 unblocked — it builds today against the local cintx checkout. Plan checker should call out that any cintx-side change merged after Phase 2 starts requires re-syncing the path dep (or completing 01-08 to switch back to git pin). Not a Phase 2 BLOCKER, but a coordination-cost item.

---

## Validation Architecture

> Required because `workflow.nyquist_validation: true` in `.planning/config.json`. (Verified.)

### Test Framework

| Property | Value |
|---|---|
| Framework | Rust standard `#[test]` (Phase 1 precedent) + `rstest 0.26.1` for parametric tests + `approx 0.5.1` for float tolerance + `pyscf-oracle` crate (PyO3-driven upstream PySCF in-process per Phase 1 ORACLE-01) |
| Config file | `Cargo.toml` workspace `[dev-dependencies]` (workspace-shared); per-crate `tests/` directories (Phase 1 established pattern, e.g., `crates/pyscf-algebra/tests/backend_matrix.rs`) |
| Quick run command | `cargo test --workspace --exclude pyscf-bench --lib -- --skip oracle_` (unit tests only, no oracle, no benches; ~30s estimate based on Phase 1 wave-merge cost) |
| Full suite command | `cargo test --workspace --features release-oracle-tests` (includes pyscf-oracle byte-identity tests; multi-minute, runs upstream PySCF via PyO3) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|---|---|---|---|---|
| GTO-01 | `pyscf.M(atom='H 0 0 0; H 0 0 1.4', basis='sto-3g')` builds Mole with 2 atoms | unit | `cargo test -p pyscf-gto --test mole_construction h2_string_form -- --exact` | ❌ Wave 0 |
| GTO-01 | All 4 (Phase-2-shipped) atom-input forms build identical Mole on H2 | unit | `cargo test -p pyscf-gto --test mole_construction atom_input_forms_consistent` | ❌ Wave 0 |
| GTO-01 (5th form) | `Mole::from_callable(...)` returns `NotYetImplemented{phase:3}` | unit | `cargo test -p pyscf-gto --test mole_construction callable_form_deferred` | ❌ Wave 0 |
| GTO-02 | All 11 basis-input forms build identical Mole on H2/cc-pVDZ | unit + parametric (`rstest`) | `cargo test -p pyscf-gto --test basis_input_forms` | ❌ Wave 0 |
| GTO-03 | All builtin basis files referenced in ALIAS resolve via `load_basis(name, "H")` | unit (parametric) | `cargo test -p pyscf-gto --test alias_resolution -- --include-ignored` | ❌ Wave 0 (the `--include-ignored` runs the slow full-ALIAS sweep; default test is small subset) |
| GTO-03 | `gto::parse(nwchem_text)` round-trip: parse(repr(parsed)) == parsed | unit | `cargo test -p pyscf-gto --test parser_roundtrip` | ❌ Wave 0 |
| GTO-04 | `mol.build()` produces `_atm` byte-identical to upstream PySCF on H2O/cc-pVDZ | **oracle** | `cargo test -p pyscf-oracle --features release-oracle-tests --test gto_byte_identity h2o_atm` | ❌ Wave 0 (depends on pyscf-oracle harness extension — Plan 03-09 in Phase 3 builds the macro; Phase 2 must precede with a stub) |
| GTO-04 | Same for `_bas`, `_env`, `ao_loc_nr`, `nao_nr` on H2O/cc-pVDZ + benzene/6-31G* + water-trimer/STO-3G | oracle (parametric) | `cargo test -p pyscf-oracle --features release-oracle-tests --test gto_byte_identity --` | ❌ Wave 0 |
| GTO-05 (loading half) | NWChem ECP file (LANL2DZ for Cu) parses to expected ParsedEcp shape | unit | `cargo test -p pyscf-gto --test ecp_parse lanl2dz_cu` | ❌ Wave 0 |
| GTO-05 (engine half) | `mol.intor("int1e_ecp")` returns `EcpEngineNotAvailable` until cintx ECP lands | unit | `cargo test -p pyscf-gto --test ecp_engine_stub returns_unavailable` | ❌ Wave 0 |
| GTO-06 | `mol.intor("int1e_ovlp_sph")` on H2/STO-3G matches analytical 2×2 overlap | smoke | `cargo test -p pyscf-gto --test intor_smoke h2_overlap` | ❌ Wave 0 |
| GTO-06 | `mol.intor(name)` for 10 representative names (`int1e_ovlp_sph`, `int1e_kin_sph`, `int1e_nuc_sph`, `int2e_sph`, `int1e_ipovlp_sph`, `int1e_ipnuc_sph`, `int3c2e_sph`, `int2c2e_sph`, `int1e_grids_sph`, `int1e_r_sph`) matches upstream within cintx tolerance | oracle | `cargo test -p pyscf-oracle --features release-oracle-tests --test intor_parity` | ❌ Wave 0 |
| GTO-06 | F-order layout preserved on output where upstream returns F-order (Pitfall 8) | unit | `cargo test -p pyscf-gto --test intor_layout f_order_preserved_int1e_ipovlp` | ❌ Wave 0 |
| GTO-07 | `eval_gto(mol, "GTOval_sph", coords)` for 1000 random grid points on H2O/cc-pVDZ matches upstream element-wise (≤1e-12) | oracle | `cargo test -p pyscf-oracle --features release-oracle-tests --test eval_gto_parity gtoval_sph_h2o` | ❌ Wave 0 |
| GTO-07 | `GTOval_sph_deriv1` matches upstream (or NotYetImplemented if planner defers) | oracle / unit | `cargo test -p pyscf-oracle ... eval_gto_deriv1` (or unit `eval_gto_deriv1_deferred`) | ❌ Wave 0 |
| GTO-08 | All ≥30 attributes present on `Mole` and have upstream-matching values for H2O | unit | `cargo test -p pyscf-gto --test attribute_floor h2o_attr_floor` | ❌ Wave 0 |
| GTO-09 | `mol.dumps()` → `Mole::loads(s)` round-trip preserves `_atm`/`_bas`/`_env` byte-identical | unit | `cargo test -p pyscf-gto --test dumps_loads roundtrip_preserves_arrays` | ❌ Wave 0 |
| GTO-10 | `mol.set_geom_("H 0 0 0; H 0 0 1.5")` mutates `_env` xyz only; `_bas` byte-identical | unit | `cargo test -p pyscf-gto --test set_geom granular_invalidation` | ❌ Wave 0 |
| GTO-11 | `mol.basis_cintx()` returns `Arc<BasisSet>` and reference-counts at >1 (zero-copy proof) | unit | `cargo test -p pyscf-gto --test cintx_zerocopy arc_count_above_one` | ❌ Wave 0 |

### Sampling Rate

- **Per task commit:** `cargo test --workspace --exclude pyscf-bench --exclude pyscf-oracle --lib -- --skip oracle_` (unit-only, no oracle PyO3 startup; ~30s target).
- **Per wave merge:** `cargo test --workspace --exclude pyscf-bench` (includes oracle harness on the small PR-CI corpus: H2O/cc-pVDZ + benzene/6-31G* + water-trimer/STO-3G; ~3-5min target).
- **Phase gate:** Full suite green AND `cargo test --features release-oracle-tests` (FMA-free profile, RAYON_NUM_THREADS=1) before `/gsd-verify-work`. Specifically: every byte-identity oracle test runs under the FMA-free `release-oracle` profile to guarantee the assertion is meaningful (Pitfall 1 + 2 mitigation, established Phase 1).

### Wave 0 Gaps

> `gsd-planner` MUST schedule these BEFORE the GTO-* tests can be authored:

- [ ] **`crates/pyscf-gto/tests/` directory created** with shared `common.rs` for fixture helpers (H2, H2O, benzene mol constructors).
- [ ] **`crates/pyscf-gto/tests/wave0_smoke.rs`** — proves `cintx_rs::SessionRequest::new(...).query_workspace()?.evaluate()` round-trips on a trivial H2-overlap fixture from inside `pyscf-gto`. **Most-important Wave 0 risk-buy-down test.**
- [ ] **`crates/pyscf-kernels/tests/wave0_cubecl_smoke.rs`** — proves a minimal `#[cube(launch_unchecked)]` vector_add kernel launches via `cubecl-cpu` from inside `pyscf-kernels` and produces correct results. **Validates Architecture Pattern 4 base case before eval_gto invests.**
- [ ] **`crates/pyscf-oracle/src/gto_macros.rs`** (or extend whatever exists from Phase 1 ORACLE-01 plan) — `oracle_check_atm!`, `oracle_check_bas!`, `oracle_check_env!` macros that use `pyo3::Python::with_gil` to drive upstream `pyscf.M(...)` and assert byte-identity. ORACLE-02 is officially Phase 3 in the roadmap, but Phase 2 cannot defer its byte-identity tests there — Phase 2 needs a minimal version. **Plan dependency: this is a Phase 2 → Phase 3 hand-off issue; Phase 2 ships the GTO-specific assertions, Phase 3 generalises.**
- [ ] **xtask `check-algebra-wall` allowlist verification** — confirm `pyscf-kernels` is permitted to import `cubecl-*` (per A5 in Assumptions Log). If not, single-line patch to the lint config.
- [ ] **`pyscf-gto/tests/intor_layout_table.rs`** — generates a per-intor-name F/C-order table by querying upstream PySCF, commits to `pyscf-gto/docs/intor_layouts.md`. Resolves Open Question §1.
- [ ] **`PYSCF_BASIS_PATH` documentation** in `CONTRIBUTING.md` (or new `docs/env-vars.md`) — mirrors Phase 1's `PYSCF_BACKEND` documentation pattern.

---

## Sources

### Primary (HIGH confidence — read directly from this repo or sibling)
- `pyscf/gto/mole.py:106-118` (M factory), `:320-415` (format_atom), `:418-466` (format_basis), `:961-980` (make_atm_env), `:984-1016` (make_bas_env), `:1029-1105` (make_env), `:1107-1160` (make_ecp_env), `:1162-1186` (tot_electrons), `:1251-1349` (dumps/loads), `:1378-1389` (nao_nr/nao_cart), `:1454-1471` (ao_loc_nr) — Apache-2.0; canonical reference for every shape decision.
- `pyscf/gto/eval_gto.py:1-248` — full `eval_gto` reference.
- `pyscf/gto/ecp.py:1-186` — ECP `type1_by_shell`, `type2_by_shell`, `so_by_shell` reference.
- `pyscf/gto/moleintor.py:41-271` (getints dispatcher), `:288+` (`_INTOR_FUNCTIONS`), `:804-820` (make_loc), `:864-870` (ascint3 name normalization).
- `pyscf/gto/basis/__init__.py:42-460` (ALIAS, GTH_ALIAS, PP_ALIAS, _BASIS_DIR), `:621-728` (load), `:730-779` (load_ecp).
- `cintx/crates/cintx-core/src/basis.rs:48-110` — `BasisSet` typed structure.
- `cintx/crates/cintx-core/src/atom.rs:62-110` — `Atom` + `NuclearModel` types.
- `cintx/crates/cintx-core/src/shell.rs:14-114` — `Shell`, `ao_per_shell`, `spinor_len`.
- `cintx/crates/cintx-compat/src/raw.rs:15-41` — `ATM_SLOTS`, `BAS_SLOTS`, `PTR_ENV_START`, all slot constants.
- `cintx/crates/cintx-rs/src/api.rs:17-454` — `SessionRequest`, `SessionQuery`, `IntegralTensor`, `EvaluationStats`.
- `crates/pyscf-core/src/{mole,basis_set,traits,error}.rs` — Phase 1 stubs (shape contract for Phase 2 fills).
- `crates/pyscf-algebra/src/{lib,client,tensor,oracle}.rs` — Phase 1 algebra surface that pyscf-kernels consumes.
- `Cargo.toml` (root) — workspace deps, `[patch.crates-io]` cintx path-dep.
- `docs/manual/Cubecl/Cubecl_multi_ compute.md` — `#[cube(launch_unchecked)]` vector_add pattern.
- `docs/manual/Cubecl/cubecl_matmul_gemm_example.md` — `client.create_tensor`, `client.empty_tensor`, `MatmulInputHandle`, `Strategy::Auto` pattern.
- `docs/manual/Cubecl/cubecl_macro_fanout_manual.md` — `#[cube]` vs `#[cube(launch)]`, comptime, generic numeric kernels guidance.
- `.planning/REQUIREMENTS.md` — GTO-01..11 + traceability.
- `.planning/ROADMAP.md` Phase 2 section + Pitfall-to-Phase Mapping (Pitfalls 8/17/18 owned by Phase 2).
- `.planning/STATE.md` — current cubecl 0.10.0 lockstep + cintx ECP coordination concerns.
- `.planning/phases/01-foundation/01-CONTEXT.md` — D-01..D-15 carried forward.
- `.planning/phases/02-gto/02-CONTEXT.md` — D-01..D-07 + Claude's Discretion + Deferred Ideas.

### Secondary (MEDIUM confidence — inferred / cross-referenced)
- `pyscf/gto/basis/parse_nwchem.py` (entry-point catalogue verified by grep) — full algorithm not read in this session; algorithm-port assumption rests on "Apache-2.0, well-tested, mature" trust + the file's straightforward parser shape.
- cintx-cubecl + xcfun-kernels precedent for kernel/facade split (referenced in CONTEXT.md canonical_refs but not directly read this session); pattern carried forward by D-04 mandate.

### Tertiary (LOW confidence — unverified)
- `cubecl 0.10.0` exact macro surface (`#[cube(launch_unchecked)]`, `Array<f64>`, `comptime` arg syntax) — relies on the documented patterns in `docs/manual/Cubecl/`. **Mitigation:** Wave 0 cubecl smoke test resolves to HIGH on day one. (Captured as A7 in Assumptions Log.)

---

## Metadata

**Confidence breakdown:**
- Standard stack: **HIGH** — every dep is already in workspace Cargo.toml or in cintx sibling (verified via `Read`).
- Architecture: **HIGH** for the port-from-upstream patterns; **MEDIUM** for the eval_gto cubecl kernel sketch (extrapolated from a vector_add example, not a kernel-precedent for grid AO eval).
- Pitfalls: **HIGH** — Pitfalls 8, 17, 18 are explicit in ROADMAP and the canonical algorithms address them.
- ECP scope split (D-06): **HIGH** — explicitly user-decided in CONTEXT.md.
- Mole attribute floor: **HIGH** — sourced from `pyscf/gto/mole.py` (in-repo).

**Research date:** 2026-05-10
**Valid until:** 2026-06-09 (30 days; pyscf upstream is mature, cintx is path-dep + locked Cargo.lock so version drift is non-issue; cubecl 0.10.0 lockstep is Phase 1's concern, not Phase 2's).

---

## RESEARCH COMPLETE — Phase 2 ready for planning
